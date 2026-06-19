#!/bin/sh

if [ -f "$HOME/.bash_profile" ]; then
  . "$HOME/.bash_profile"
elif [ -f "$HOME/.zprofile" ]; then
  . "$HOME/.zprofile"
fi

if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  PATH="/opt/homebrew/opt/ruby/bin:$PATH"
fi

bundle exec jekyll serve --lsi
