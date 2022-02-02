# single-char comment for simplicity

# makeshift changes:
# - prefilled array syntax

var foo;     # global variable, uninitialized
var foo = 1; # global variable with init

# Variable with array initialization (`[ ]` returns pointer to list)
var foo = [ 1, 1, 2, 3, 5, 8 ];

# Constants are compile-time only
const bar = 10;
const bam = bar + 5; # but can be used at compile time after being declared

fn main() {
  var x = 10; # local variables can be declared like global ones
  var y = square(x); # functions can be called before declaration (and with any number of arguments)
  
  sum(foo, 6);
}

fn square(x) {
  return x * x;
}

fn sum(array, len) {
  var  s= 0;
  while(len > 0) {
    len = len - 1;
    s = s + array[len]; # anything can be accessed as a pointer by using array operators
  }
  return s;
}

fn strStuff() {
  var str = { 30, 30, 37, 0 };
  var len = strlen(str);
}

fn strlen(str) {
  var len = 0;
  while(str@(len) != 0) {
    len = len + 1;
  }
  return len;
}

fn str16len(str) {
  var len = 0;
  while(str[len] != 0) {
    len = len + 1;
  }
  return len;
}

fn decider(x) {
  if(x > 0) {
    return x * 2;
  }
  else {
    return -x;
  }
}

## Binary Operators
# >, <, >=, <=, !=, == are relational operators
# +, -, *, /, %        are arithmetic operators
# &, |, ^              are bitwise operartors 
# and, or              are logic operators (short circuit)
#

# Unary Operators 
# ~           is bitwise inversion
# !           is logic inversion 
# <<, >>, >>> are shift operators (shift 1)
# &           is the "addressof" operator
#

# Other Operators
# val[x]            will dereference val + sizeof(word) * x as a word
# [ a, b, c, ... ]  will create an array of words
# val{x}            will dereference val + x as  byte
# { a, b, c, ... }  will create an array of bytes

## Syntax constructs
# while(cond) { }
# break;
# continue;
# return;
# return val;
# if(cond) { }
# if(cond) { } else { }
# var name;
# var name = value;
# var array = { item, item, … };
# const name = value;
# const array = { item, item, … };
# fncall(), fncall(x), fncall(x, y, …)
# x = result;

## Properties
# - no operator precedence, use parens
# - use before declare allowed, might warn
# - variables are 16 bit 
# - c style expressions
# 
