module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
