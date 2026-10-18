module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %reduction = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %reduction iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
