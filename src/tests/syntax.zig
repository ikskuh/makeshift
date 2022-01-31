const std = @import("std");
const mott = @import("../main.zig");

test "all syntax constructs" {
    const demo_source = @embedFile("syntax.mott");

    var ast = try mott.Parser.parse(std.testing.allocator, demo_source, "src/tests/syntax.mott");
    defer ast.deinit();
}

fn expectError(source: []const u8, comptime errors: []const []const u8) !void {
    var diangostics = mott.Diagnostics.init(std.testing.allocator);
    defer diangostics.deinit();

    var ast = try mott.Parser.parse(std.testing.allocator, source, null);
    defer ast.deinit();

    var env = try mott.SemanticAnalysis.check(std.testing.allocator, &diangostics, ast);
    defer env.deinit();

    try std.testing.expectEqual(errors.len, diangostics.errors.items.len);
    for (errors) |err, i| {
        try std.testing.expectEqualStrings(err, diangostics.errors.items[i].message);
    }
}

fn expectNoError(source: []const u8) !void {
    var diangostics = mott.Diagnostics.init(std.testing.allocator);
    defer diangostics.deinit();

    var ast = try mott.Parser.parse(std.testing.allocator, source, null);
    defer ast.deinit();

    var env = try mott.SemanticAnalysis.check(std.testing.allocator, &diangostics, ast);
    defer env.deinit();

    try std.testing.expect(!diangostics.hasErrors());
}

test "empty file" {
    try expectNoError(
        \\
    );
}

test "variable declaration" {
    try expectNoError(
        \\var x;
    );
}

test "variable declaration with number initialization" {
    try expectNoError(
        \\var x = 10;
    );
}

test "const declaration with number initialization" {
    try expectNoError(
        \\const x = 10;
    );
}

test "lhs assignments" {
    try expectNoError(
        \\const x = 10;
        \\fn f() {
        \\  x = 10;
        \\}
    );
    try expectNoError(
        \\const x = 10;
        \\fn f() {
        \\  x[3 + 5] = 10;
        \\}
    );
    try expectNoError(
        \\const x = 10;
        \\fn f() {
        \\  x@1 = 10;
        \\}
    );
    try expectNoError(
        \\const x = 10;
        \\fn f() {
        \\  x@(0 + 2) = 10;
        \\}
    );
}

test "undeclared var in global" {
    try expectError(
        \\var global = other;
    , &.{"The function or variable `other` does not exist."});
}

test "undeclared var in const" {
    try expectError(
        \\const global = other;
    , &.{"The function or variable `other` does not exist."});
}

test "undeclared var in fn" {
    try expectError(
        \\fn foo() { other; }
    , &.{"The function or variable `other` does not exist."});
}

test "duplicate symbols" {
    try expectError(
        \\fn foo() { }
        \\fn foo() { }
    , &.{"A symbol with the name `foo` already exists."});

    try expectError(
        \\var foo;
        \\fn foo() { }
    , &.{"A symbol with the name `foo` already exists."});

    try expectError(
        \\const foo = 0;
        \\fn foo() { }
    , &.{"A symbol with the name `foo` already exists."});

    try expectError(
        \\fn foo() { }
        \\var foo;
    , &.{"A symbol with the name `foo` already exists."});

    try expectError(
        \\const foo = 0;
        \\fn foo() { }
    , &.{"A symbol with the name `foo` already exists."});
}

test "non-lvalue assignment" {
    try expectError(
        \\var x = 0;
        \\fn f() {
        \\  x+1 = 0;
        \\}
    , &.{"Left hand side of an assignment must be an lvalue."});

    try expectError(
        \\var x = 0;
        \\fn f() {
        \\  -x = 0;
        \\}
    , &.{"Left hand side of an assignment must be an lvalue."});
}

test "scoped locals" {
    try expectError(
        \\fn foo() { 
        \\  var x = 1;
        \\  if(1) {
        \\    var y = 2;
        \\    y = x;
        \\  }
        \\  y();
        \\}
    , &.{"The function or variable `y` does not exist."});
}
