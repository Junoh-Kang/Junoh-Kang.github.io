#!/usr/bin/env ruby
# frozen_string_literal: true

ROOT = File.expand_path("..", __dir__)

def read_repo_file(path)
  File.read(File.join(ROOT, path))
end

def collect_failure(failures, message)
  failures << message
end

failures = []

index = read_repo_file("blog/index.html")
featured_section = index[/<section class="[^"]*featured-posts-list[^"]*">.*?<\/section>/m]
latest_section = index[/<section class="[^"]*latest-posts-list[^"]*">.*?<\/section>/m]

collect_failure(failures, "blog index should define a featured posts block") unless index.include?("assign featured_posts = site.posts")
collect_failure(failures, "featured section should be titled Featured") unless index.include?("<h2>Featured</h2>")
collect_failure(failures, "latest section should be titled Latest Posts") unless index.include?("<h2>Latest Posts</h2>")
collect_failure(failures, "blog pagination should show more than five posts") unless index.include?("per_page: 10")
collect_failure(failures, "featured posts should render as an archive-style table") unless featured_section&.include?("table table-sm table-borderless")
collect_failure(failures, "latest posts should render as an archive-style table") unless latest_section&.include?("table table-sm table-borderless")
collect_failure(failures, "latest posts should be explicitly left aligned") unless latest_section&.include?('<section class="latest-posts-list text-left">') || latest_section&.include?('class="table table-sm table-borderless text-left"')
collect_failure(failures, "featured posts should not use card UI") if index.include?("card hoverable")
collect_failure(failures, "featured posts should not use feed-style post-list UI") if featured_section&.include?('<ul class="post-list">')
collect_failure(failures, "featured posts should not use a separate category/date column") if featured_section&.include?('<th scope="row">')
collect_failure(failures, "featured posts should show descriptions") unless featured_section&.include?("{{ post.description }}")
collect_failure(failures, "featured title link should include an inline category label") unless featured_section&.include?("[{{ category_label }}]") && featured_section&.include?("{{ post.title }}</a>")
collect_failure(failures, "featured posts should show tags") unless featured_section&.include?("{% for tag in post.tags %}")
collect_failure(failures, "featured tags should not be right aligned") if featured_section&.include?("ml-auto text-right")
collect_failure(failures, "featured description and tags should share the smaller metadata line") unless featured_section&.include?('<p class="text-muted small mb-0">{{ post.description }}')
if featured_section
  tag_position = featured_section.index("{% for tag in post.tags %}")
  description_position = featured_section.index("{{ post.description }}")
  collect_failure(failures, "featured tags should follow the description") unless tag_position && description_position && tag_position > description_position
end
collect_failure(failures, "featured category label should route to the post, not the category page") if featured_section&.include?("prepend: '/blog/category/'")
collect_failure(failures, "latest posts should use compact date rows") unless latest_section&.include?('{{ post.date | date: "%b %-d, %Y" }}')
collect_failure(failures, "latest titles should include inline category labels") unless latest_section&.include?("[{{ category_label }}]") && latest_section&.include?("{{ post.title }}</a>")
collect_failure(failures, "latest category label should route to the post, not the category page") if latest_section&.include?("prepend: '/blog/category/'")

featured_paths = [
  "_posts/2026-06-19-dl: Post-Training of Modern LLMs.md",
  "_posts/2026-03-12-dl: One-step Generation in the Post Diffusion Era.md",
  "_posts/2023-09-03-dl : Understanding Diffusion Models in Two Perspectives.md"
]

featured_paths.each do |path|
  collect_failure(failures, "#{path} should be featured") unless read_repo_file(path).include?("featured: true")
end

sample_paths = [
  "_posts/ref/code.md",
  "_posts/ref/distill.md"
]

sample_paths.each do |path|
  collect_failure(failures, "#{path} should not be featured") if read_repo_file(path).include?("featured: true")
end

if failures.empty?
  puts "blog featured layout checks passed"
else
  warn failures.map { |failure| "- #{failure}" }.join("\n")
  exit 1
end
