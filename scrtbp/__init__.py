from llvmlite import binding

binding.set_option("tmp", "-non-global-value-max-name-size=2048")
