# 🔩 Makeshift - A dead-simple programming language

Makeshift is a programming language that was designed to bootstrap the [Ashet Home Computer](https://ashet.computer/). It is meant to run on 16 bit machines and is currently implemented with that restriction.

Check out some example source:

```cs
const text = { 48, 48, 55, 0 };

fn strlen(ptr) {
  var len = 0;
  while(ptr@(len)) {
    len = len + 1;
  }
  return len;
}

fn main() {
  return strlen(text);
}
```

## Design goals

- Make a programming language that is suitable for 16 bit machines
- Make a programming language that can be bootstrapped from assembly.
- Don't care too much about type or memory safety
- Well-known syntax constructs (C style)

## Project Status

**EXPERIMENTAL**

The parser and interpreter for Makeshift is implemented already, but the compiler and (optional) optimizer is still missing.


