load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["@heir//:license"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "@heir//tests:test_utilities",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:mlir-runner",
        "@llvm-project//mlir:libmlir_c_runner_utils.so",
        "@llvm-project//mlir:libmlir_runner_utils.so",
    ],
    driver = "@heir//tests:run_lit.sh",
    test_file_exts = ["mlir"],
)
