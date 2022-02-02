# comment

var var1;

var var2 = 10;

const const1 = 20;

fn fn1 () {}
fn fn2 (a) {}
fn fn3 (a,b) {}
fn fn3 (a,b,c) {}
fn fn4 (a,b,c) {
  {}
  {}
}
fn fn5 (a,b,c) {
  break;
}
fn fn6 (a,b,c) {
  continue;
}
fn fn7 (a,b,c) {
  return;
}
fn fn8 (a,b,c) {
  return 30;
}
fn fn9 (a,b,c) {
  if(30) { }
}
fn fn10 (a,b,c) {
  if(30) { } else { }
}
fn fn11 (a,b,c) {
  while(30) { }
}

fn fn12 () {
  var local;
}

fn fn13 () {
  var local = 200;
}

fn fn14() {
  300;
}

fn fn15() {
   varname = 400;
}

fn fn16() {
  if(1) {
    if(2) {

    }
    if(3) {
      if(4) {

      }
    }
  }
}

fn fn17() {
  (1);
  ((2));
  (((3)));
  ((((4))));
  (((((5)))));
}

fn fn18() {
  1 + 2;
  1 + 2 + 3;
  1 + 2 + 3 + 4;
}

fn fn19() {
  a + b;
  a - b;
  a * b;
  a / b;
  a % b;
  a & b;
  a | b;
  a ^ b;
  a > b;
  a < b;
  a >= b;
  a <= b;
  a == b;
  a != b;
  a and b;
  a or b;
}

fn fn20() {
  -a;
}

fn fn21() {
  --a;
  !-a;
  -!a;
}

fn fn22() {
  - a;
  ~ a;
  ! a;
  & a;
  << a;
  >> a;
  >>> a;
}

fn fn23() {
  a[0];
}

fn fn23() {
  a@0;
  a@(0);
  a@(1 + 2);
  a@(1 + 2 + 3);
}

fn fn24() {
  a();
}

fn fn25() {
  a(1);
}

fn fn26() {
  a(1,2);
  a(1,2,3);
  a(1,2,3,4);
  a(1,2,3,4,5);
}

fn fn27() {
  -a();
}

fn fn28() {
  (-a)();
}

fn fn29() {
  a = [];
  a = [1];
  a = [1,2];
  a = [1,2,3];
  a = [1,2,3,4];
}

fn fn30() {
  a = {};
  a = {1};
  a = {1,2};
  a = {1,2,3};
  a = {1,2,3,4};
}

fn fn31() {
  f( a@31 );
  f( a@(x+y) );
}

fn fn32() {
  a    = 1;
  a@0  = 2;
  a[0] = 3;
}
