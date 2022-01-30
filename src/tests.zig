const std = @import("std");
const mott = @import("main.zig");

test "all syntax constructs" {
    const demo_source = @embedFile("tests/syntax.mott");

    var ast = try mott.Parser.parse(std.testing.allocator, demo_source, "src/tests/syntax.mott");
    defer ast.deinit();
}
