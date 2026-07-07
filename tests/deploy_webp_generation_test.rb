#!/usr/bin/env ruby
# frozen_string_literal: true

ROOT = File.expand_path("..", __dir__)

def read_repo_file(path)
  File.read(File.join(ROOT, path))
end

failures = []
workflow = read_repo_file(".github/workflows/deploy.yml")

unless workflow.include?("sudo apt-get install -y imagemagick webp")
  failures << "deploy workflow should install ImageMagick and WebP tooling before the Jekyll build"
end

%w[480 800 1400].each do |width|
  next if workflow.include?("_site/assets/img/profile1-#{width}.webp")

  failures << "deploy workflow should verify profile1-#{width}.webp exists before deployment"
end

if failures.empty?
  puts "deploy WebP generation checks passed"
else
  warn failures.map { |failure| "- #{failure}" }.join("\n")
  exit 1
end
