#!/usr/bin/env ruby

require "date"
require "fileutils"
require "json"
require "pathname"
require "uri"
require "yaml"

module CvExport
  class InvalidVisibility < StandardError; end
  class ExportError < StandardError; end

  class Exporter
    # Matches the homepage accent (navy #1f4e79).
    SITE_NAVY = "rgb(31, 78, 121)"
    ARXIV_ID = %r{arxiv\.org/(?:abs|pdf)/([^/?#]+?)(?:\.pdf)?(?:[?#].*)?\z}

    def initialize(root = File.expand_path(__dir__))
      @root = File.expand_path(root)
    end

    # Structured public data for the Astro site: public items only, visibility stripped.
    def render_site_data
      validate!

      data = {
        "profile" => site_profile,
        "sections" => public_sections.map do |section_id, section|
          compact_deep(
            {
              "id" => section_id,
              "title" => section["title"],
              "note" => section["note"],
              "items" => public_items(section).map { |item| item.reject { |key, _| key == "visibility" } }
            }
          )
        end
      }

      JSON.pretty_generate(data) + "\n"
    end

    def write_site_data
      write_file(output_path("site_data"), render_site_data)
    end

    def render_rendercv_data
      validate!

      generated_header + rendercv_model.to_yaml.sub(/\A---\n/, "")
    end

    def write_rendercv_data
      write_file(output_path("rendercv_data"), render_rendercv_data)
    end

    def build_pdf
      rendercv_data_path = write_rendercv_data
      pdf_path = output_path("pdf")
      FileUtils.mkdir_p(File.dirname(pdf_path))
      rendercv_data_dir = File.dirname(rendercv_data_path)
      pdf_relative_path = Pathname.new(pdf_path).relative_path_from(Pathname.new(rendercv_data_dir)).to_s

      success = system(
        *rendercv_command,
        "render",
        File.basename(rendercv_data_path),
        "--pdf-path",
        pdf_relative_path,
        "--dont-generate-markdown",
        "--dont-generate-html",
        "--dont-generate-png",
        chdir: rendercv_data_dir
      )
      raise ExportError, "rendercv render failed" unless success
      raise ExportError, "rendercv did not write #{pdf_path}" unless File.exist?(pdf_path)

      FileUtils.rm_rf(File.join(rendercv_data_dir, "rendercv_output"))

      pdf_path
    end

    def write_all
      write_site_data if config.fetch("outputs").key?("site_data")
      build_pdf
    end

    def publish_site
      write_all
      copy_outputs(config.fetch("publish_site", {}))
    end

    private

    def copy_outputs(publish_config)
      publish_config.each do |name, destination|
        source = output_path(name)
        target = resolve_path(destination)
        FileUtils.mkdir_p(File.dirname(target))
        FileUtils.cp(source, target)
      end

      publish_config.values.map { |path| resolve_path(path) }
    end

    def site_profile
      compact_deep(
        {
          "name" => profile["name"],
          "headline" => profile["headline"],
          "affiliations" => public_only(profile.fetch("affiliations", [])),
          "contacts" => public_only(profile.fetch("contacts", []))
        }
      )
    end

    def public_only(items)
      items.select { |item| item["visibility"] == public_visibility }
           .map { |item| item.reject { |key, _| key == "visibility" } }
    end

    attr_reader :root

    def config
      @config ||= load_yaml("cv.yml")
    end

    def profile
      @profile ||= load_yaml(config.fetch("profile"))
    end

    def sections
      @sections ||= config.fetch("sections").map do |section_id|
        [section_id, load_yaml(File.join("sections", "#{section_id}.yml"))]
      end
    end

    def section_by_id
      @section_by_id ||= sections.to_h
    end

    def public_sections
      sections.select { |_section_id, section| public_items(section).any? }
    end

    def public_items(section)
      section.fetch("items", []).select { |item| item["visibility"] == public_visibility }
    end

    def public_visibility
      config.dig("rules", "public_exports_include") || "public"
    end

    def allowed_visibility
      config.dig("rules", "visibility_values") || %w[public private archive]
    end

    def validate!
      validate_visibility!("profile affiliations", profile.fetch("affiliations", []))
      validate_visibility!("profile contacts", profile.fetch("contacts", []))
      validate_visibility!("profile postal_address", [profile["postal_address"]].compact)

      sections.each do |section_id, section|
        validate_visibility!(section_id, section.fetch("items", []))
      end
    end

    def validate_visibility!(label, items)
      items.each do |item|
        visibility = item["visibility"]
        next if allowed_visibility.include?(visibility)

        raise InvalidVisibility, "#{label} has invalid visibility #{visibility.inspect}; allowed: #{allowed_visibility.join(", ")}"
      end
    end

    def load_yaml(relative_path)
      YAML.load_file(File.join(root, relative_path))
    end

    def output_path(name)
      value = config.fetch("outputs").fetch(name)
      resolve_path(value)
    end

    def resolve_path(path)
      return path if PathnameAbsolute.absolute?(path)

      File.join(root, path)
    end

    def write_file(path, content)
      FileUtils.mkdir_p(File.dirname(path))
      File.write(path, content)
      path
    end

    def generated_header
      "# Generated from #{File.join(root, "cv.yml")}. Do not edit directly.\n"
    end

    def rendercv_model
      compact_deep(
        {
          "cv" => {
            "name" => profile.fetch("name"),
            "headline" => rendercv_headline,
            "email" => public_contact_value("email"),
            "custom_connections" => rendercv_custom_connections,
            "sections" => rendercv_sections
          },
          "design" => {
            "theme" => rendercv_config["theme"] || "engineeringresumes",
            "colors" => {
              "name" => "rgb(0, 0, 0)",
              "connections" => "rgb(0, 0, 0)",
              "section_titles" => SITE_NAVY,
              "links" => "rgb(0, 0, 0)"
            },
            "links" => {
              "underline" => false
            },
            "page" => {
              "show_top_note" => !rendercv_last_updated.nil?
            },
            "sections" => {
              "allow_page_break" => false,
              "space_between_regular_entries" => "0.8em"
            },
            "entries" => {
              "highlights" => {
                "space_above" => "0.18cm"
              }
            },
            "header" => {
              "connections" => {
                "show_icons" => true,
                "display_urls_instead_of_usernames" => false
              }
            },
            "templates" => {
              "top_note" => rendercv_top_note,
              "normal_entry" => {
                "date_and_location_column" => "LOCATION\nYEAR_LABEL\nDATE"
              }
            }
          }
        }
      )
    end

    def rendercv_config
      @rendercv_config ||= config.fetch("rendercv", {})
    end

    def rendercv_headline
      return rendercv_config["headline"] if rendercv_config.key?("headline")

      profile["headline"]
    end

    def rendercv_last_updated
      value = rendercv_config["last_updated"]
      return Date.today.strftime("%B %Y") if value == "auto"

      value
    end

    def rendercv_top_note
      return unless rendercv_last_updated

      "Last updated: #{rendercv_last_updated}"
    end

    def public_location
      profile.fetch("affiliations", [])
        .find { |affiliation| affiliation["visibility"] == public_visibility && affiliation["location"] }
        &.fetch("location")
    end

    def public_contacts
      @public_contacts ||= profile.fetch("contacts", []).select { |contact| contact["visibility"] == public_visibility }
    end

    def public_contact_value(type)
      public_contacts.find { |contact| contact["type"] == type }&.fetch("value")
    end

    def public_contact_url(type)
      public_contacts.find { |contact| contact["type"] == type }&.fetch("url")
    end

    def rendercv_custom_connections
      public_contacts.sort_by { |contact| rendercv_contact_priority(contact["type"]) }.filter_map do |contact|
        case contact["type"]
        when "website"
          rendercv_custom_connection("globe", rendercv_contact_display(contact), contact)
        when "google_scholar"
          rendercv_custom_connection("graduation-cap", rendercv_contact_display(contact), contact)
        when "linkedin"
          rendercv_custom_connection("linkedin", rendercv_contact_display(contact), contact)
        when "github"
          rendercv_custom_connection("github", rendercv_contact_display(contact), contact)
        end
      end
    end

    def rendercv_contact_priority(type)
      {
        "website" => 0,
        "github" => 1,
        "linkedin" => 2,
        "google_scholar" => 3
      }.fetch(type, 99)
    end

    def rendercv_contact_display(contact)
      case contact["type"]
      when "website"
        compact_url_label(contact["url"])
      when "google_scholar"
        google_scholar_user_id(contact["url"]) || compact_url_label(contact["url"])
      when "github", "linkedin"
        contact["label"] || trailing_path_segment(contact["url"]) || compact_url_label(contact["url"])
      else
        contact["label"] || compact_url_label(contact["url"]) || contact["value"] || contact.fetch("type")
      end
    end

    def compact_url_label(url)
      return unless url && !url.empty?

      uri = URI.parse(url)
      host = uri.host
      return stripped_url_label(url) unless host

      host = host.sub(/\Awww\./, "")
      path = uri.path.to_s
      label = host.dup
      label += path unless path.empty? || path == "/"
      label += "?#{uri.query}" if uri.query && !uri.query.empty?
      label
    rescue URI::InvalidURIError
      stripped_url_label(url)
    end

    def stripped_url_label(url)
      url.to_s.sub(/\Ahttps?:\/\//, "").sub(/\Awww\./, "").sub(/\/\z/, "")
    end

    def google_scholar_user_id(url)
      query = URI.parse(url).query
      return unless query

      URI.decode_www_form(query).find { |key, _value| key == "user" }&.last
    rescue URI::InvalidURIError
      nil
    end

    def trailing_path_segment(url)
      path = URI.parse(url).path
      path.split("/").reject(&:empty?).last
    rescue URI::InvalidURIError
      nil
    end

    def rendercv_custom_connection(icon, label, contact)
      {
        "fontawesome_icon" => icon,
        "placeholder" => label,
        "url" => contact["url"]
      }
    end

    def rendercv_sections
      rendercv_section_ids.each_with_object({}) do |section_id, output|
        section = section_by_id.fetch(section_id)
        items = public_items(section)
        next if items.empty?

        output[rendercv_section_title(section_id, section)] = rendercv_entries(section_id, items)
      end
    end

    def rendercv_section_ids
      rendercv_config["sections"] || config.fetch("sections")
    end

    def rendercv_section_title(section_id, section)
      case section_id
      when "publications"
        "Selected Publications"
      when "experience"
        "Experience"
      else
        section.fetch("title")
      end
    end

    def rendercv_entries(section_id, items)
      case section_id
      when "publications"
        items.map { |item| publication_to_rendercv(item) }
      when "experience"
        items.map { |item| experience_to_rendercv(item) }
      when "honors"
        items.map { |item| honor_to_rendercv(item) }
      when "service"
        items.map { |item| service_to_rendercv(item) }
      when "teaching"
        items.map { |item| teaching_to_rendercv(item) }
      else
        items.map { |item| timeline_item_to_rendercv(item) }
      end
    end

    def publication_to_rendercv(item)
      {
        "name" => markdown_link(item.fetch("title"), publication_url(item)),
        "summary" => item.fetch("authors").map { |author| rendercv_author(author) }.join(", "),
        "date" => item.fetch("venue")
      }
    end

    def experience_to_rendercv(item)
      {
        "company" => markdown_link(item["organization"] || item["institution"], item["url"]),
        "position" => item.fetch("title"),
        "location" => item["location"],
        "date" => item["date"],
        "highlights" => rendercv_details(item["details"])
      }
    end

    def honor_to_rendercv(item)
      {
        "name" => markdown_link(item.fetch("title"), item["url"]),
        "year_label" => item.fetch("year").to_s
      }
    end

    def service_to_rendercv(item)
      {
        "label" => item.fetch("role"),
        "details" => item.fetch("venues").map { |venue| venue.sub(/\s+\d{4}\z/, "") }.uniq.join(", ")
      }
    end

    def teaching_to_rendercv(item)
      {
        "company" => item.fetch("institution"),
        "position" => "#{item.fetch("title")}, #{item.fetch("course")}",
        "date" => item.fetch("date")
      }
    end

    def timeline_item_to_rendercv(item)
      {
        "name" => markdown_link(item.fetch("title"), item["url"]),
        "location" => item["location"],
        "date" => item["date"],
        "summary" => item["institution"] || item["organization"],
        "highlights" => rendercv_details(item["details"])
      }
    end

    # alphaXiv (built from the arXiv link) first, since it opens fast; then the published
    # PDF. The homepage uses the same rule for its Paper link.
    def publication_url(item)
      links = item.fetch("links", [])
      arxiv = links.find { |l| l["label"] == "arXiv" }&.fetch("url")
      if arxiv
        id = arxiv[ARXIV_ID, 1]
        return id ? "https://www.alphaxiv.org/abs/#{id}" : arxiv
      end
      links.find { |l| l["label"] == "Paper" }&.fetch("url")
    end

    def markdown_link(label, url)
      return label.to_s unless url && !url.empty?

      "[#{label}](#{url})"
    end

    def rendercv_details(details)
      Array(details).map { |detail| rendercv_detail(detail) }
    end

    def rendercv_detail(detail)
      return detail.to_s unless detail.is_a?(Hash)

      markdown_link(detail_label(detail), detail["url"])
    end

    def detail_label(detail)
      detail.fetch("label")
    end

    def rendercv_text(value)
      value.to_s.gsub("*", "\\*")
    end

    def rendercv_author(author)
      author = author.to_s
      self_name = profile.fetch("name")
      return rendercv_text(author) unless author.start_with?(self_name)

      suffix = author.delete_prefix(self_name)
      "#underline[#{rendercv_text(self_name)}]#{rendercv_text(suffix)}"
    end

    def rendercv_command
      if ENV["RENDERCV"] && !ENV["RENDERCV"].empty?
        [ENV["RENDERCV"]]
      elsif command_available?("rendercv")
        ["rendercv"]
      elsif command_available?("uv")
        ["uv", "run", "--with", "rendercv[full]", "rendercv"]
      else
        ["rendercv"]
      end
    end

    def command_available?(name)
      ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).any? do |directory|
        File.executable?(File.join(directory, name))
      end
    end

    def compact_deep(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, child), output|
          compacted = compact_deep(child)
          output[key] = compacted unless empty_compacted_value?(compacted)
        end
      when Array
        value.map { |child| compact_deep(child) }.reject { |child| empty_compacted_value?(child) }
      else
        value
      end
    end

    def empty_compacted_value?(value)
      value.nil? || (value.respond_to?(:empty?) && value.empty?)
    end
  end

  module PathnameAbsolute
    module_function

    def absolute?(path)
      path.start_with?("/")
    end
  end
end

if __FILE__ == $PROGRAM_NAME
  usage = "usage: ruby export.rb [rendercv-data|site-data|pdf|all|publish-site] [root]"
  command = ARGV.shift || "all"
  root = ARGV.shift || File.expand_path(__dir__)
  abort usage unless ARGV.empty?

  exporter = CvExport::Exporter.new(root)

  path = case command
         when "rendercv-data"
           exporter.write_rendercv_data
         when "site-data"
           exporter.write_site_data
         when "pdf"
           exporter.build_pdf
         when "all"
           exporter.write_all
           "all outputs"
         when "publish-site"
           exporter.publish_site.join(", ")
         else
           abort usage
         end

  puts "Wrote #{path}"
end
