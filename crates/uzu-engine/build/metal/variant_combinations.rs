/// Enumerate satisfying selections in Cartesian-product order while pruning partial assignments.
pub fn constrained_combinations(
    domain_lengths: &[usize],
    mut could_satisfy: impl FnMut(&[Option<usize>], bool) -> bool,
) -> Vec<Box<[usize]>> {
    let complete = domain_lengths.is_empty();
    let mut selection = vec![None; domain_lengths.len()];
    let viable = could_satisfy(&selection, complete);
    if !viable {
        return Vec::new();
    }
    if complete {
        return vec![Box::new([])];
    }

    // Binding small domains first exposes format and boolean constraints before
    // descending into larger tile families. The final sort restores the original Cartesian order.
    let mut axis_order: Vec<_> = (0..domain_lengths.len()).collect();
    axis_order.sort_by_key(|&axis| domain_lengths[axis]);

    let mut combinations = Vec::new();
    let first_axis = axis_order[0];
    let mut stack = vec![(first_axis, 0..domain_lengths[first_axis])];
    while let Some((axis, values)) = stack.last_mut() {
        let axis = *axis;
        let value = values.next();
        let Some(value) = value else {
            selection[axis] = None;
            stack.pop();
            continue;
        };

        selection[axis] = Some(value);
        let next_axis = axis_order.get(stack.len()).copied();
        let viable = could_satisfy(&selection, next_axis.is_none());
        if !viable {
            continue;
        }
        let Some(next_axis) = next_axis else {
            combinations.push(selection.iter().map(|value| value.unwrap()).collect());
            continue;
        };

        // Each frame owns one axis, clearing its binding when its values are exhausted.
        stack.push((next_axis, 0..domain_lengths[next_axis]));
    }

    combinations.sort_unstable();
    combinations
}
