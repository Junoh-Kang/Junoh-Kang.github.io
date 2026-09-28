ROOT := .

.PHONY: test rendercv-data blog-data site-data pdf all publish-site

test:
	ruby "$(ROOT)/test_export.rb"

rendercv-data:
	ruby "$(ROOT)/export.rb" rendercv-data "$(ROOT)"

blog-data:
	ruby "$(ROOT)/export.rb" blog-data "$(ROOT)"

site-data:
	ruby "$(ROOT)/export.rb" site-data "$(ROOT)"

pdf:
	ruby "$(ROOT)/export.rb" pdf "$(ROOT)"

all:
	ruby "$(ROOT)/export.rb" all "$(ROOT)"

publish-site:
	ruby "$(ROOT)/export.rb" publish-site "$(ROOT)"
