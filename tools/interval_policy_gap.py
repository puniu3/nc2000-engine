import math


def oriented_cells(matrix, side):
    assert side in (0, 1)
    rows, cols = map(len, matrix['actions'])
    assert rows > 0 and cols > 0 and len(matrix['cells']) == rows * cols
    for cell in matrix['cells']:
        assert math.isfinite(cell['lo']) and math.isfinite(cell['hi'])
        assert 0 <= cell['lo'] <= cell['hi'] <= 1
    if side == 0:
        return [[matrix['cells'][i * cols + j] for j in range(cols)] for i in range(rows)]
    return [[{'lo': 1 - matrix['cells'][i * cols + j]['hi'],
              'hi': 1 - matrix['cells'][i * cols + j]['lo']} for i in range(rows)] for j in range(cols)]


def certify(matrix, side, policy):
    cells = oriented_cells(matrix, side)
    rows, cols = len(cells), len(cells[0])
    assert len(policy) == rows and all(math.isfinite(p) and p >= 0 for p in policy)
    assert abs(math.fsum(policy) - 1) < 1e-12
    candidates = []
    for opponent_reply in range(cols):
        differences = []
        for alternative in range(rows):
            for policy_reply in range(cols):
                terms = []
                for action in range(rows):
                    for reply in range(cols):
                        coefficient = float(action == alternative and reply == opponent_reply)
                        if reply == policy_reply:
                            coefficient -= policy[action]
                        terms.append(coefficient * cells[action][reply]['hi' if coefficient >= 0 else 'lo'])
                differences.append(math.fsum(terms))
        upper = max(0.0, max(differences))
        candidates.append((upper, opponent_reply, differences))
    upper, reply, differences = min(candidates)
    return {'schema': 'nc2000-correlated-policy-gap-v1', 'side': side, 'policy': policy,
            'opponent_reply': reply, 'difference_upper_bounds': differences,
            'loss_upper': min(1.0, upper + 1e-10), 'numerical_envelope': 1e-10}


def verify(matrix, certificate):
    cells = oriented_cells(matrix, certificate['side'])
    rows, cols = len(cells), len(cells[0])
    p = certificate['policy']
    assert len(p) == rows and all(math.isfinite(x) and x >= 0 for x in p)
    assert abs(math.fsum(p) - 1) < 1e-12
    opponent_reply = certificate['opponent_reply']
    assert 0 <= opponent_reply < cols
    values = []
    for i in range(rows):
        for j in range(cols):
            if j != opponent_reply:
                value = cells[i][opponent_reply]['hi'] - math.fsum(p[a] * cells[a][j]['lo'] for a in range(rows))
            else:
                value = (1 - p[i]) * cells[i][j]['hi'] - math.fsum(p[a] * cells[a][j]['lo'] for a in range(rows) if a != i)
            values.append(value)
    assert len(certificate['difference_upper_bounds']) == len(values)
    assert all(abs(a - b) < 1e-12 for a, b in zip(values, certificate['difference_upper_bounds']))
    assert max(0.0, max(values)) - 1e-12 <= certificate['loss_upper'] <= 1
    return True
