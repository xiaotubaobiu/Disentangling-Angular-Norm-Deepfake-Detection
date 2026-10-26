from deepfake_detector import launcher as l


def test_shared_python_binary_does_not_make_global_environment_project_env(tmp_path):
    assert hasattr(l,'needs_reexec'), 'environment-prefix gate not implemented'
    global_prefix=tmp_path/'base';env=tmp_path/'project-env';global_prefix.mkdir();env.mkdir()
    shortcut=tmp_path/'venv';shortcut.symlink_to(env,target_is_directory=True)
    assert l.needs_reexec(global_prefix,shortcut)
    assert not l.needs_reexec(env,shortcut)
