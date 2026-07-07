#!/usr/bin/env ruby
# frozen_string_literal: true

ROOT = File.expand_path("..", __dir__)

def read_repo_file(path)
  File.read(File.join(ROOT, path))
end

failures = []
figure_include = read_repo_file("_includes/figure.html")

unless figure_include.include?("$('.responsive-img-srcset').remove(); this.src=")
  failures << "figure image fallback should reset the image src to the original asset when responsive WebP loading fails"
end

if failures.empty?
  puts "profile image fallback checks passed"
else
  warn failures.map { |failure| "- #{failure}" }.join("\n")
  exit 1
end
