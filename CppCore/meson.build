_incdirs += include_directories(['include'])

# --------------------- Executable

tiny_cli = executable('tiny_cli', 'tiny_cli.cpp',
                       dependencies: _deps,
                       include_directories: _incdirs,
                       cpp_args: _args,
                       link_with: _linkto,
                       install: false)

if get_option('with_tests')
    test_deps = _deps
    test_deps += dependency(
      'Catch2',
      method: 'cmake',
      modules: ['Catch2::Catch2WithMain'],
    )
    test_args = _args
    test_array = [#
      ['pdist_euc_test', 'pdist_euc_test.cpp', '/CppCore/tests'],
      ['pdist_seuc_test', 'pdist_seuc_test.cpp', '/CppCore/tests'],
      ['pdist_sqeuc_test', 'pdist_sqeuc_test.cpp', '/CppCore/tests'],
      ['squareform_test', 'squareform_test.cpp', ''],
      ['util_test', 'util_test.cpp', ''],
    ]
    foreach test : test_array
      test(test.get(0),
           executable(test.get(0),
              sources : ['tests/'+test.get(1)],
              dependencies : test_deps,
              include_directories: _incdirs,
              cpp_args: test_args,
              link_with: _linkto,
                     ),
            workdir : meson.source_root() + test.get(2)
          )
    endforeach
endif
