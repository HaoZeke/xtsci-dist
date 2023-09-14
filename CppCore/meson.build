_incdirs += include_directories(['include'])

# --------------------- Executable

tiny_cli = executable('tiny_cli', 'tiny_cli.cpp',
                       dependencies: _deps,
                       include_directories: _incdirs,
                       cpp_args: _args,
                       link_with: _linkto,
                       install: false)
