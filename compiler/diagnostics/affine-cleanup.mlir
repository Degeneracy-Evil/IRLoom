// Source-independent integer reduction; no Polygeist frontend is involved.
module {
  func.func @same_init(%n: index) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %scale = arith.constant 1000 : i64
    %s:2 = affine.for %i = 0 to %n iter_args(%a = %zero, %b = %zero) -> (i64, i64) {
      %aa = arith.addi %a, %one : i64
      %bb = arith.addi %b, %two : i64
      affine.yield %aa, %bb : i64, i64
    }
    %scaled = arith.muli %s#0, %scale : i64
    %result = arith.addi %scaled, %s#1 : i64
    return %result : i64
  }
  func.func @distinct_init(%n: index) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %scale = arith.constant 1000 : i64
    %s:2 = affine.for %i = 0 to %n iter_args(%a = %one, %b = %two) -> (i64, i64) {
      %aa = arith.addi %a, %one : i64
      %bb = arith.addi %b, %two : i64
      affine.yield %aa, %bb : i64, i64
    }
    %scaled = arith.muli %s#0, %scale : i64
    %result = arith.addi %scaled, %s#1 : i64
    return %result : i64
  }
  func.func @bound_alias(%n: index) -> i64 {
    %one = arith.constant 1 : index
    %s = affine.for %i = 0 to %n iter_args(%a = %n) -> index {
      %aa = arith.addi %a, %one : index
      affine.yield %aa : index
    }
    %result = arith.index_cast %s : index to i64
    return %result : i64
  }
  func.func @constant_odd() -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %scale = arith.constant 1000 : i64
    %s:2 = affine.for %i = 0 to 3 iter_args(%a = %zero, %b = %zero) -> (i64, i64) {
      %aa = arith.addi %a, %one : i64
      %bb = arith.addi %b, %two : i64
      affine.yield %aa, %bb : i64, i64
    }
    %scaled = arith.muli %s#0, %scale : i64
    %result = arith.addi %scaled, %s#1 : i64
    return %result : i64
  }
  func.func @constant_even() -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %scale = arith.constant 1000 : i64
    %s:2 = affine.for %i = 0 to 4 iter_args(%a = %zero, %b = %zero) -> (i64, i64) {
      %aa = arith.addi %a, %one : i64
      %bb = arith.addi %b, %two : i64
      affine.yield %aa, %bb : i64, i64
    }
    %scaled = arith.muli %s#0, %scale : i64
    %result = arith.addi %scaled, %s#1 : i64
    return %result : i64
  }
}
