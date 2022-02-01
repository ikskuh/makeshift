const std = @import("std");
const mott = @import("../main.zig");

fn expectError(source: []const u8, comptime errors: []const []const u8) !void {
    var diagnostics = mott.Diagnostics.init(std.testing.allocator);
    defer diagnostics.deinit();

    var ast = try mott.Parser.parse(std.testing.allocator, source, null);
    defer ast.deinit();

    var env = try mott.SemanticAnalysis.check(std.testing.allocator, &diagnostics, ast);
    defer env.deinit();

    try std.testing.expectEqual(errors.len, diagnostics.errors.items.len);
    for (errors) |err, i| {
        try std.testing.expectEqualStrings(err, diagnostics.errors.items[i].message);
    }
}

fn compileExpectNoError(source: []const u8) !void {
    var diagnostics = mott.Diagnostics.init(std.testing.allocator);
    defer diagnostics.deinit();

    var ast = try mott.Parser.parse(std.testing.allocator, source, null);
    defer ast.deinit();

    var env = try mott.SemanticAnalysis.check(std.testing.allocator, &diagnostics, ast);
    defer env.deinit();

    try std.testing.expect(!diagnostics.hasErrors());
}

fn runExpectNoError(expected_result: u16, source: []const u8) !void {
    var diagnostics = mott.Diagnostics.init(std.testing.allocator);
    defer diagnostics.deinit();

    var ast = try mott.Parser.parse(std.testing.allocator, source, null);
    defer ast.deinit();

    var env = try mott.SemanticAnalysis.check(std.testing.allocator, &diagnostics, ast);
    defer env.deinit();

    try std.testing.expect(!diagnostics.hasErrors());

    var interpreter = mott.Interpreter.init(std.testing.allocator, &env, ast);
    const actual_result = try interpreter.run(&.{});

    try std.testing.expectEqual(expected_result, actual_result);
}

test "initialize variable" {
    try compileExpectNoError(
        \\var x = 1;
        \\fn main() {}
    );
}

test "initialize variable with backref" {
    try compileExpectNoError(
        \\const x = 1337;
        \\const y = x;
    );
}

test "comptime invocation variable with backref" {
    try compileExpectNoError(
        \\fn add(a, b) {
        \\  return a + b;
        \\}
        \\const ten = 10;
        \\const twen = 20;
        \\const comptime = add(ten, twen);
    );
}

test "storing local variables must work at comptime" {
    try compileExpectNoError(
        \\fn storeLocal(a) {
        \\  var b = a + 1;
        \\  return b - 1;
        \\}
        \\const stored_local = storeLocal(3);
    );
}

test "successful detection of indirect runtime data read" {
    try expectError(
        \\var foo;
        \\var addr_of_foo = &foo;
        \\fn storeGlobal(a) {
        \\  addr_of_foo[0] = a;
        \\}
        \\const side_effect = storeGlobal(10);
    , &.{"Cannot evaluate constant expression. Read from runtime data."});
}

test "successful detection of indirection runtime data write" {
    try expectError(
        \\var foo;
        \\var addr_of_foo = &foo;
        \\fn storeGlobal(a) {
        \\  (&addr_of_foo)[0] = a;
        \\}
        \\const side_effect = storeGlobal(10);
    , &.{"Cannot evaluate constant expression. Write to runtime data."});
}

test "indirect calling" {
    try expectError(
        \\fn foo(a) {
        \\  return 2 * a;
        \\}
        \\const foo_ref = foo;
        \\const res = foo_ref(1);
    , &.{"The callee `foo_ref` is not a function. This might be a mistake."});
}

test "init global word array" {
    try runExpectNoError(29,
        \\const data = [ 13, 9, 7 ];
        \\fn main() {
        \\  var i = 3;
        \\  var sum = 0;
        \\  while(i > 0) {
        \\    i = i - 1;
        \\    sum = sum + data[i];
        \\  }
        \\  return sum;
        \\}
    );
}

test "init local word array" {
    try runExpectNoError(29,
        \\fn main() {
        \\  var before = 0;
        \\  var data = [ 13, 9, 7 ];
        \\  var after = 0;
        \\  var i = 3;
        \\  var sum = 0;
        \\  while(i > 0) {
        \\    i = i - 1;
        \\    sum = sum + data[i];
        \\  }
        \\  return sum - before - after; # check if we have off-by-one errors
        \\}
    );
}

test "init global byte array" {
    try runExpectNoError(29,
        \\const data = { 13, 9, 7 };
        \\fn main() {
        \\  var i = 3;
        \\  var sum = 0;
        \\  while(i > 0) {
        \\    i = i - 1;
        \\    sum = sum + data@(i);
        \\  }
        \\  return sum;
        \\}
    );
}

test "init local byte array" {
    try runExpectNoError(29,
        \\fn main() {
        \\  var before = 0;
        \\  var data = { 13, 9, 7 };
        \\  var after = 0;
        \\  var i = 3;
        \\  var sum = 0;
        \\  while(i > 0) {
        \\    i = i - 1;
        \\    sum = sum + data@(i);
        \\  }
        \\  return sum - before - after; # check if we have off-by-one errors
        \\}
    );
}

test "strlen implementation" {
    try runExpectNoError(3,
        \\const text = { 48, 48, 55, 0 };
        \\fn strlen(ptr) {
        \\  var len = 0;
        \\  while(ptr@(len)) {
        \\    len = len + 1;
        \\  }
        \\  return len;
        \\}
        \\fn main() {
        \\  return strlen(text);
        \\}
    );
}

test "if" {
    try runExpectNoError(1,
        \\fn decider(x) {
        \\  if(x > 100) {
        \\    return 1;
        \\  }
        \\  return 2;
        \\}
        \\fn main() {
        \\  return decider(1010);
        \\}
    );
    try runExpectNoError(2,
        \\fn decider(x) {
        \\  if(x > 100) {
        \\    return 1;
        \\  }
        \\  return 2;
        \\}
        \\fn main() {
        \\  return decider(10);
        \\}
    );
}

test "if-else" {
    try runExpectNoError(1,
        \\fn decider(x) {
        \\  if(x > 100) {
        \\    return 1;
        \\  } else {
        \\    return 2;
        \\  }
        \\}
        \\fn main() {
        \\  return decider(1010);
        \\}
    );
    try runExpectNoError(2,
        \\fn decider(x) {
        \\  if(x > 100) {
        \\    return 1;
        \\  } else {
        \\    return 2;
        \\  }
        \\}
        \\fn main() {
        \\  return decider(10);
        \\}
    );
}
