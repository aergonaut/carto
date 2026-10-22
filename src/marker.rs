//! Drawing reference markers (Xs) at given points.

/// Marks an X centered on each `(col, row)` pixel, with diagonals spanning
/// `length` pixels. Parts falling outside the image are clipped.
pub fn mask(centers: &[(i64, i64)], rows: usize, cols: usize, length: u32) -> Vec<bool> {
    let mut marks = vec![false; rows * cols];
    let half = (length.max(1) as i64 - 1) / 2;
    for &(c, r) in centers {
        for k in -half..=half {
            for (pc, pr) in [(c + k, r + k), (c + k, r - k)] {
                if (0..cols as i64).contains(&pc) && (0..rows as i64).contains(&pr) {
                    marks[pr as usize * cols + pc as usize] = true;
                }
            }
        }
    }
    marks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_clipped_x() {
        let m = mask(&[(1, 1), (4, 0)], 3, 5, 3);
        let rows: Vec<String> = m
            .chunks(5)
            .map(|r| r.iter().map(|&b| if b { '#' } else { '.' }).collect())
            .collect();
        assert_eq!(rows, ["#.#.#", ".#.#.", "#.#.."]);
    }
}
