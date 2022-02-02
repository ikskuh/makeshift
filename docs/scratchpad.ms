
const string = "Hello, World!";

fn strlen(ptr) {
  var len = 0;
  while(ptr@(len)) {
    len = len + 1;
  }
  return len;
}

fn main(argc, argv) {
  return strlen("hi");
}