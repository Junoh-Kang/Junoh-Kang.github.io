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
config = read_repo_file("_config.yml")
featured_section = index[/<section class="[^"]*featured-posts-list[^"]*">.*?<\/section>/m]
latest_section = index[/<section class="[^"]*latest-posts-list[^"]*">.*?<\/section>/m]
latest_posts_table = latest_section[/<div class="table-responsive">.*?<\/div>/m]
browse_section = index[/<div class="[^"]*blog-browse-links[^"]*">.*?<\/div>/m]

collect_failure(failures, "blog index should define a featured posts block") unless index.include?("assign featured_posts = site.posts")
collect_failure(failures, "featured section should be titled Featured") unless index.include?("<h2>Featured</h2>")
collect_failure(failures, "all posts section should be titled All Posts") unless index.match?(%r{<h2[^>]*>All Posts</h2>})
collect_failure(failures, "blog pagination should show more than five posts") unless index.include?("per_page: 10")
featured_position = index.index('<section class="featured-posts-list">')
pagination_position = index.index("{% include pagination.html %}")
browse_position = index.index("blog-browse-links")
top_browse_position = index.index("tag-category-list")
collect_failure(failures, "category/tag navigation should not float above featured posts") if top_browse_position && featured_position && top_browse_position < featured_position
collect_failure(failures, "category/tag navigation should live below the All Posts heading") unless latest_section&.include?("</h2>\n    {% if site.display_tags.size > 0 or site.display_categories.size > 0 %}")
collect_failure(failures, "category/tag navigation should appear before the latest posts table") unless browse_position && latest_section&.index("blog-browse-links") && latest_section&.index('<div class="table-responsive">') && latest_section.index("blog-browse-links") < latest_section.index('<div class="table-responsive">')
collect_failure(failures, "category/tag navigation should not live below pagination") if browse_position && pagination_position && browse_position > pagination_position
collect_failure(failures, "All Posts heading should not use a right-side flex browse row") if latest_section&.include?("d-flex flex-column flex-md-row") || latest_section&.include?("justify-content-between")
collect_failure(failures, "category/tag browsing should use smaller muted text") unless browse_section&.include?("small") && browse_section&.include?("text-muted")
collect_failure(failures, "category/tag browsing should not be centered") if browse_section&.include?("text-center")
collect_failure(failures, "category/tag browsing should stay left aligned under All Posts") unless browse_section&.include?("text-left")
collect_failure(failures, "category browsing should keep label and links inline") unless browse_section&.include?("Category:") && browse_section&.include?("{% for category in site.display_categories %}")
collect_failure(failures, "tag browsing should keep label and links inline when display tags are configured") unless browse_section&.include?("Tag:") && browse_section&.include?("{% for tag in site.display_tags %}")
collect_failure(failures, "category browsing should render as the first heading metadata line") unless browse_section&.include?('<span class="blog-browse-category d-block">')
collect_failure(failures, "tag browsing should render as the second heading metadata line") unless browse_section&.include?('<span class="blog-browse-tags d-block">')
collect_failure(failures, "two-line browsing should not use desktop-only left margin between category and tag") if browse_section&.include?("ml-md-3")
if browse_section
  category_label_position = browse_section.index("Category:")
  tag_label_position = browse_section.index("Tag:")
  collect_failure(failures, "tag browsing should appear after the category links") unless category_label_position && tag_label_position && tag_label_position > category_label_position
end
collect_failure(failures, "browse labels should not force line breaks before links") if browse_section&.include?("<br")
collect_failure(failures, "blog index should configure displayed tags for heading browsing") unless config.include?("display_tags: [generative, llm, rl, video, test-time-scaling, time-series, statistics]")
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
collect_failure(failures, "latest posts should use compact date rows") unless latest_posts_table&.include?('{{ post.date | date: "%b %-d, %Y" }}')
collect_failure(failures, "latest titles should include inline category labels") unless latest_posts_table&.include?("[{{ category_label }}]") && latest_posts_table&.include?("{{ post.title }}</a>")
collect_failure(failures, "latest category label should route to the post, not the category page") if latest_posts_table&.include?("prepend: '/blog/category/'")

featured_paths = [
  "_posts/2026-06-19-Post-Training of Modern LLMs.md",
  "_posts/2026-03-12-One-step Generation in the Post Diffusion Era.md",
  "_posts/2023-09-03-Understanding Diffusion Models in Two Perspectives.md"
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
