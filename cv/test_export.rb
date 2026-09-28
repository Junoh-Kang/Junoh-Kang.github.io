require "fileutils"
require "json"
require "open3"
require "tmpdir"
require "yaml"
require "date"
require "minitest/autorun"

require_relative "export"

class CvExportTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir("cv-export-test")
    FileUtils.mkdir_p(File.join(@tmpdir, "sections"))
    FileUtils.mkdir_p(File.join(@tmpdir, "templates"))

    write("cv.yml", <<~YAML)
      name: Test Person
      profile: profile.yml
      sections:
        - education
        - publications
        - experience
        - honors
        - references
      outputs:
        rendercv_data: build/rendercv.yml
        pdf: build/Test_Person_CV.pdf
        blog_data: build/cv.yml
        site_data: build/site.json
      rendercv:
        theme: classic
        headline:
        last_updated: June 2026
        sections:
          - education
          - experience
          - publications
          - honors
      rules:
        visibility_values: [public, private, archive]
        public_exports_include: public
    YAML

    write("profile.yml", <<~YAML)
      name: Test Person
      headline: Test headline
      affiliations:
        - name: Public University
          location: Test City
          visibility: public
      contacts:
        - type: email
          value: public@example.com
          visibility: public
        - type: email
          value: private@example.com
          visibility: private
        - type: website
          label: Homepage
          url: https://example.com
          visibility: public
        - type: github
          label: TestGitHub
          url: https://github.com/TestGitHub
          visibility: public
        - type: linkedin
          label: TestLinkedIn
          url: https://www.linkedin.com/in/testlinkedin
          visibility: public
        - type: google_scholar
          label: Google Scholar
          url: https://scholar.google.com/citations?user=testuser&hl=en
          visibility: public
    YAML

    write("sections/education.yml", <<~YAML)
      title: Education
      items:
        - title: Public Degree
          institution: Public University
          date: 2026
          details:
            - label: "Advisor: Prof. Public"
              url: https://example.com/advisor
          visibility: public
        - title: Archived Degree
          institution: Old University
          date: 2010
          visibility: archive
    YAML

    write("sections/publications.yml", <<~YAML)
      title: Publications
      note: "(*) denotes equal contribution."
      items:
        - id: person2026public
          title: Public Paper
          authors:
            - Test Person*
            - Coauthor
          venue: arXiv 2026
          links:
            - label: arXiv
              url: https://example.com/paper
            - label: Code
              url: https://example.com/code
          award: Best Paper Award
          visibility: public
        - title: Private Paper
          authors:
            - Test Person
          venue: Private
          visibility: private
    YAML

    write("sections/experience.yml", <<~YAML)
      title: Work Experience
      items:
        - title: Research Intern
          organization: Research Lab
          location: Seoul, Korea
          date: Summer 2026
          details:
            - Built a model.
          visibility: public
    YAML

    write("sections/honors.yml", <<~YAML)
      title: Honors and Awards
      items:
        - title: Public Award
          year: 2025
          url: https://example.com/award
          visibility: public
        - title: Private Award
          year: 2024
          visibility: private
    YAML

    write("sections/references.yml", <<~YAML)
      title: References
      items:
        - name: Private Reference
          email: private-reference@example.com
          visibility: private
    YAML
  end

  def teardown
    FileUtils.remove_entry(@tmpdir)
  end

  def test_render_rendercv_data_includes_public_entries_and_excludes_non_public_entries
    data = CvExport::Exporter.new(@tmpdir).render_rendercv_data

    assert_includes data, "Test Person"
    assert_includes data, "public@example.com"
    assert_includes data, "Public Degree"
    assert_includes data, "Public Paper"
    refute_includes data, "private@example.com"
    refute_includes data, "Archived Degree"
    refute_includes data, "Private Paper"
    refute_includes data, "Private Reference"
  end

  def test_write_rendercv_data_uses_configured_output_path
    path = CvExport::Exporter.new(@tmpdir).write_rendercv_data

    assert_equal File.join(@tmpdir, "build", "rendercv.yml"), path
    assert File.exist?(path), "expected RenderCV output to be written"
    assert_includes File.read(path), "Public Degree"
  end

  def test_public_config_places_publications_after_experience
    root = File.expand_path(__dir__)
    data = YAML.load(CvExport::Exporter.new(root).render_rendercv_data)

    assert_equal(
      ["Education", "Experience", "Selected Publications", "Honors and Awards", "Academic Services", "Teaching"],
      data.dig("cv", "sections").keys
    )
  end

  def test_archived_current_research_preserves_advisor_and_collaborator
    root = File.expand_path(__dir__)
    entry = YAML.load_file(File.join(root, "sections", "current_research.yml")).fetch("items").first

    assert_equal "archive", entry["visibility"]
    assert_equal "Advised by Prof. Bohyung Han, in collaboration with Prof. Kiseop Lee", entry["institution"]
  end

  def test_public_outputs_exclude_archived_current_research
    root = File.expand_path(__dir__)
    exporter = CvExport::Exporter.new(root)
    rendercv_data = YAML.load(exporter.render_rendercv_data)
    blog_data = YAML.load(exporter.render_blog_data)

    refute rendercv_data.dig("cv", "sections").key?("Current Research")
    refute blog_data.any? { |section| section["title"] == "Current Research" }
  end

  def test_render_rendercv_data_uses_pdf_order_theme_and_public_entries_only
    data = YAML.load(CvExport::Exporter.new(@tmpdir).render_rendercv_data)

    assert_equal "Test Person", data.dig("cv", "name")
    refute data.dig("cv", "headline")
    refute data.dig("cv", "location")
    assert_equal "public@example.com", data.dig("cv", "email")
    refute data.dig("cv", "website")
    refute data.dig("cv", "social_networks")
    assert_equal(
      [
        { "fontawesome_icon" => "globe", "placeholder" => "example.com", "url" => "https://example.com" },
        { "fontawesome_icon" => "github", "placeholder" => "TestGitHub", "url" => "https://github.com/TestGitHub" },
        { "fontawesome_icon" => "linkedin", "placeholder" => "TestLinkedIn", "url" => "https://www.linkedin.com/in/testlinkedin" },
        { "fontawesome_icon" => "graduation-cap", "placeholder" => "testuser", "url" => "https://scholar.google.com/citations?user=testuser&hl=en" },
      ],
      data.dig("cv", "custom_connections")
    )
    assert_equal "classic", data.dig("design", "theme")
    assert_equal false, data.dig("design", "links", "underline")
    assert_equal "rgb(0, 0, 0)", data.dig("design", "colors", "links")
    assert_equal "rgb(0, 0, 0)", data.dig("design", "colors", "name")
    assert_equal "rgb(0, 0, 0)", data.dig("design", "colors", "connections")
    assert_equal "rgb(31, 78, 121)", data.dig("design", "colors", "section_titles")
    assert_equal "LOCATION\nYEAR_LABEL\nDATE", data.dig("design", "templates", "normal_entry", "date_and_location_column")
    assert_equal true, data.dig("design", "page", "show_top_note")
    assert_equal "Last updated: June 2026", data.dig("design", "templates", "top_note")
    assert_equal false, data.dig("design", "sections", "allow_page_break")
    assert_equal "0.8em", data.dig("design", "sections", "space_between_regular_entries")
    assert_equal "0.18cm", data.dig("design", "entries", "highlights", "space_above")
    assert_equal true, data.dig("design", "header", "connections", "show_icons")
    assert_equal false, data.dig("design", "header", "connections", "display_urls_instead_of_usernames")
    assert_equal ["Education", "Experience", "Selected Publications", "Honors and Awards"], data.dig("cv", "sections").keys
    assert_equal "Public Degree", data.dig("cv", "sections", "Education", 0, "name")
    assert_equal(
      ["[Advisor: Prof. Public](https://example.com/advisor)"],
      data.dig("cv", "sections", "Education", 0, "highlights")
    )
    assert_equal "Research Lab", data.dig("cv", "sections", "Experience", 0, "company")
    assert_equal "[Public Paper](https://example.com/paper)", data.dig("cv", "sections", "Selected Publications", 0, "name")
    assert_equal "#underline[Test Person]\\*, Coauthor", data.dig("cv", "sections", "Selected Publications", 0, "summary")
    assert_equal "arXiv 2026", data.dig("cv", "sections", "Selected Publications", 0, "date")
    assert_equal "[Public Award](https://example.com/award)", data.dig("cv", "sections", "Honors and Awards", 0, "name")
    assert_equal "2025", data.dig("cv", "sections", "Honors and Awards", 0, "year_label")
    refute data.dig("cv", "sections", "Honors and Awards", 0, "date")
    refute_includes data.to_s, "Archived Degree"
    refute_includes data.to_s, "Private Paper"
    refute_includes data.to_s, "Private Award"
    refute_includes data.to_s, "Private Reference"
  end

  def test_render_rendercv_data_expands_auto_last_updated_to_build_month
    path = File.join(@tmpdir, "cv.yml")
    config = File.read(path).sub("last_updated: June 2026", "last_updated: auto")
    File.write(path, config)

    original_today = Date.method(:today)
    Date.define_singleton_method(:today) { Date.new(2026, 7, 8) }
    begin
      data = YAML.load(CvExport::Exporter.new(@tmpdir).render_rendercv_data)

      assert_equal true, data.dig("design", "page", "show_top_note")
      assert_equal "Last updated: July 2026", data.dig("design", "templates", "top_note")
    ensure
      Date.define_singleton_method(:today) { original_today.call }
    end
  end

  def test_render_blog_data_uses_public_entries_only
    data = YAML.load(CvExport::Exporter.new(@tmpdir).render_blog_data)

    assert_equal ["Education", "Publications", "Work Experience", "Honors and Awards"], data.map { |section| section["title"] }
    assert_equal "time_table", data.fetch(0).fetch("type")
    assert_equal "Public Degree", data.fetch(0).fetch("contents").fetch(0).fetch("title")
    assert_equal(
      [%(<a href="https://example.com/advisor">Advisor: Prof. Public</a>)],
      data.fetch(0).fetch("contents").fetch(0).fetch("description")
    )
    refute_includes data.to_s, "Archived Degree"
    refute_includes data.to_s, "Private Paper"
    refute_includes data.to_s, "Private Reference"
  end

  def test_write_blog_data_uses_configured_output_path
    path = CvExport::Exporter.new(@tmpdir).write_blog_data

    assert_equal File.join(@tmpdir, "build", "cv.yml"), path
    assert File.exist?(path), "expected blog data output to be written"
    assert_includes File.read(path), "Public Degree"
  end

  def test_render_site_data_uses_public_entries_only_without_visibility
    data = JSON.parse(CvExport::Exporter.new(@tmpdir).render_site_data)

    assert_equal %w[education publications experience honors], data.fetch("sections").map { |section| section["id"] }
    assert_equal "Test Person", data.fetch("profile").fetch("name")
    assert_equal ["public@example.com"], data.fetch("profile").fetch("contacts").map { |c| c["value"] }.compact
    refute_includes data.to_s, "Archived Degree"
    refute_includes data.to_s, "Private Paper"
    refute_includes data.to_s, "Private Award"
    refute_includes data.to_s, "Private Reference"
    refute_includes data.to_s, "private@example.com"
    refute_includes JSON.generate(data), "visibility"
  end

  def test_render_site_data_keeps_publication_id_and_links
    data = JSON.parse(CvExport::Exporter.new(@tmpdir).render_site_data)
    paper = data.fetch("sections").find { |s| s["id"] == "publications" }.fetch("items").fetch(0)

    assert_equal "person2026public", paper.fetch("id")
    assert_equal [{ "label" => "arXiv", "url" => "https://example.com/paper" }, { "label" => "Code", "url" => "https://example.com/code" }], paper.fetch("links")
    assert_equal "Best Paper Award", paper.fetch("award")
    assert_equal "(*) denotes equal contribution.", data.fetch("sections").find { |s| s["id"] == "publications" }.fetch("note")
  end

  def test_publication_title_link_prefers_alphaxiv_then_paper
    exporter = CvExport::Exporter.new(@tmpdir)
    both = { "links" => [{ "label" => "Paper", "url" => "https://p" }, { "label" => "arXiv", "url" => "https://arxiv.org/abs/2405.11473" }] }
    arxiv_pdf = { "links" => [{ "label" => "arXiv", "url" => "https://arxiv.org/pdf/2306.00001v2" }] }
    paper_only = { "links" => [{ "label" => "Paper", "url" => "https://p" }] }
    none = { "links" => [{ "label" => "Code", "url" => "https://c" }] }

    assert_equal "https://www.alphaxiv.org/abs/2405.11473", exporter.send(:publication_url, both)
    assert_equal "https://www.alphaxiv.org/abs/2306.00001v2", exporter.send(:publication_url, arxiv_pdf)
    assert_equal "https://p", exporter.send(:publication_url, paper_only)
    assert_nil exporter.send(:publication_url, none)
  end

  def test_write_site_data_uses_configured_output_path
    path = CvExport::Exporter.new(@tmpdir).write_site_data

    assert_equal File.join(@tmpdir, "build", "site.json"), path
    assert_includes File.read(path), "Public Degree"
  end

  def test_invalid_visibility_is_rejected
    write("sections/education.yml", <<~YAML)
      title: Education
      items:
        - title: Bad Entry
          visibility: secret
    YAML

    error = assert_raises(CvExport::InvalidVisibility) do
      CvExport::Exporter.new(@tmpdir).render_rendercv_data
    end

    assert_includes error.message, "secret"
  end

  def test_cli_rejects_removed_named_config_argument
    script = File.join(File.expand_path(__dir__), "export.rb")

    _stdout, stderr, status = Open3.capture3(
      "ruby", script, "rendercv-data", @tmpdir, "legacy.yml"
    )

    refute status.success?
    assert_includes stderr, "usage:"
  end

  def test_build_pdf_invokes_rendercv_with_generated_input_and_configured_pdf_path
    fake_rendercv = File.join(@tmpdir, "fake-rendercv")
    log = File.join(@tmpdir, "rendercv-args.log")
    write("fake-rendercv", <<~SH)
      #!/bin/sh
      printf '%s\\n' "$@" > "$RENDERCV_TEST_LOG"
      while [ "$#" -gt 0 ]; do
        if [ "$1" = "--pdf-path" ]; then
          shift
          mkdir -p rendercv_output
          printf 'TYPST' > rendercv_output/Test_Person_CV.typ
          mkdir -p "$(dirname "$1")"
          printf 'PDF' > "$1"
          exit 0
        fi
        shift
      done
      exit 1
    SH
    File.chmod(0o755, fake_rendercv)

    old_command = ENV["RENDERCV"]
    old_log = ENV["RENDERCV_TEST_LOG"]
    ENV["RENDERCV"] = fake_rendercv
    ENV["RENDERCV_TEST_LOG"] = log

    path = CvExport::Exporter.new(@tmpdir).build_pdf

    assert_equal File.join(@tmpdir, "build", "Test_Person_CV.pdf"), path
    assert_equal "PDF", File.read(path)
    refute_path_exists File.join(@tmpdir, "build", "rendercv_output")
    assert_includes File.read(log), "render"
    assert_includes File.read(log), "rendercv.yml"
    assert_includes File.read(log), "--pdf-path"
    assert_includes File.read(log), "Test_Person_CV.pdf"
  ensure
    ENV["RENDERCV"] = old_command
    ENV["RENDERCV_TEST_LOG"] = old_log
  end

  def test_build_pdf_raises_when_rendercv_does_not_write_pdf
    fake_rendercv = File.join(@tmpdir, "fake-rendercv")
    write("fake-rendercv", <<~SH)
      #!/bin/sh
      exit 0
    SH
    File.chmod(0o755, fake_rendercv)

    old_command = ENV["RENDERCV"]
    ENV["RENDERCV"] = fake_rendercv

    error = assert_raises(CvExport::ExportError) do
      CvExport::Exporter.new(@tmpdir).build_pdf
    end

    assert_includes error.message, "did not write"
  ensure
    ENV["RENDERCV"] = old_command
  end

  private

  def write(relative_path, content)
    path = File.join(@tmpdir, relative_path)
    FileUtils.mkdir_p(File.dirname(path))
    File.write(path, content)
  end
end
