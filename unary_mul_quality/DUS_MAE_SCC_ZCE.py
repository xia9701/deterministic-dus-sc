

import random
import numpy as np
from scipy.stats.qmc import Halton
from scipy.stats.qmc import Sobol

def set_spine_linewidth(ax, linewidth=2):
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)


def generate_random_bit():
    """Generate a random bit, '1' or '0', with equal probability."""
    return random.choice(['0', '1'])


def generate_random_8bit():
    """Generate an 8-bit random binary number, excluding all zeros."""
    while True:
        binary_str = ''.join(generate_random_bit() for _ in range(8))
        if not all(bit == '0' for bit in binary_str):
            return binary_str


def binary_to_decimal(binary_str):
    return int(binary_str, 2)


def binary_to_random_bitstream(binary_str, lengthN):
    """Generate a random bitstream and compare with binary input."""
    random_bitstream = ''
    for _ in range(lengthN):
        random_8bit_str = generate_random_8bit()
        if int(binary_str, 2) > int(random_8bit_str, 2):
            random_bitstream += '1'
        else:
            random_bitstream += '0'
    return random_bitstream


def compute_scc(X, Y):
    assert len(X) == len(Y)
    L = len(X)
    a = sum(1 for x, y in zip(X, Y) if x == '1' and y == '1')
    b = sum(1 for x, y in zip(X, Y) if x == '1' and y == '0')
    c = sum(1 for x, y in zip(X, Y) if x == '0' and y == '1')
    d = sum(1 for x, y in zip(X, Y) if x == '0' and y == '0')

    numerator = a * d - b * c
    if a * d > b * c:
        denominator = L * min(a + b, a + c) - (a + b) * (a + c)
    else:
        denominator = (a + b) * (a + c) - L * max(a - d, 0)

    if denominator == 0:
        return 0.0
    return abs(numerator / denominator)


def generate_255bit_stream_N3(binary_str, A, lengthN):
    """Generate a random bitstream and compare with binary input."""
    normalized_value = int(binary_str, 2) / 255 * (lengthN - 1)
    random_bitstream = ''
    for num in A:
        if normalized_value > num:
            random_bitstream += '1'
        else:
            random_bitstream += '0'
    return random_bitstream


def generate_255bit_stream_N3_UGMEE_sequentialA(binary_str, A, random_bitstream_N2A, lengthN):
    """Generate a masked bitstream with sequential A consumption."""
    normalized_value = int(binary_str, 2) / 255 * (lengthN - 1)
    random_bitstream = ''
    A_index = 0  # 独立推进 A 的指针

    for i in range(lengthN):
        if random_bitstream_N2A[i] == '1':
            if normalized_value > A[A_index]:
                random_bitstream += '1'
            else:
                random_bitstream += '0'
            A_index += 1
        else:
            random_bitstream += '0'

    return random_bitstream


BIT_WIDTH_LFSR = 12
LFSR_MASK_10BIT = (1 << BIT_WIDTH_LFSR) - 1  

def generate_master_lfsr_sequence(taps, seed):
    sr = seed & LFSR_MASK_10BIT
    if sr == 0:
        sr = 1

    seq = []
    visited = set()
    while sr not in visited:
        visited.add(sr)
        seq.append(sr)

        bit = 0
        for t in taps:
            bit ^= (sr >> (t - 1)) & 1
        sr = ((sr << 1) & LFSR_MASK_10BIT) | bit

    return seq



LFSR_TAPS_A = [10, 7]
LFSR_TAPS_B = [10, 3]

MASTER_LFSR_A_SEQ = generate_master_lfsr_sequence(LFSR_TAPS_A, seed=0b1010010101)
MASTER_LFSR_B_SEQ = generate_master_lfsr_sequence(LFSR_TAPS_B, seed=0b1100100110)

PERIOD_LFSR_A = len(MASTER_LFSR_A_SEQ)
PERIOD_LFSR_B = len(MASTER_LFSR_B_SEQ)

def get_lfsr_thresholds_for_N(lengthN, seq):

    period = len(seq)
    A = []
    for i in range(lengthN):
        sr = seq[i % period]
        a_val = (sr * lengthN) // (1 << BIT_WIDTH_LFSR)
        if a_val >= lengthN:
            a_val = lengthN - 1
        A.append(a_val)
    return A

def generate_255bit_stream_LFSR(binary_str, A, lengthN):
    normalized_value = int(binary_str, 2) / 255 * (lengthN - 1)

    return ''.join('1' if normalized_value > a else '0' for a in A)


def F_mul(S1, S2):
    """Perform bitwise AND on two bitstreams."""
    return ''.join('1' if S1[i] == '1' and S2[i] == '1' else '0' for i in range(len(S1)))


def compute_stream_mul_error(streamA, streamB, ref_val, lengthN):

    result = F_mul(streamA, streamB)
    error = abs(ref_val - result.count('1') / (lengthN - 1))
    return error


def compute_zce(X, Y):
    assert len(X) == len(Y)
    L = len(X)

    # 计算 a, b, c
    a = sum(1 for x, y in zip(X, Y) if x == '1' and y == '1')
    b = sum(1 for x, y in zip(X, Y) if x == '1' and y == '0')
    c = sum(1 for x, y in zip(X, Y) if x == '0' and y == '1')

    # 实际误差 Δ
    P_X = (a + b) / L
    P_Y = (a + c) / L
    P_XY = a / L
    delta = P_XY - (P_X * P_Y)

    # 理想误差 Δ0 = 量化后的乘积减去精确乘积
    raw_product = P_X * P_Y
    quantized_product = round(raw_product * L) / L
    delta_0 = quantized_product - raw_product

    # 避免除以零
    if delta == 0:
        return 0.0

    zce = abs(delta * (1 - abs(delta_0 / delta)))
    return zce


def simulate_and_collect_errors(simulations=10, lengthN=1023):
    """Run simulations and collect BtoS & MUL errors with k MAC operations."""

    NumberAA = []
    NumberBB = []
    ref_number = []

    # MUL
    error_MUL_LFSR1_LFSR2 = []
    error_MUL_H1_H2 = []
    error_MUL_S1_S2 = []
    error_MUL_uMUL = []
    error_MUL_N2_S2 = []
    error_MUL_Random = []



    scc_Random_list = []
    scc_LFSR_list = []
    scc_Halton_list = []
    scc_Sobol_list = []
    scc_uMUL_list = []
    scc_DUS_list = []

    zce_Random_list = []
    zce_LFSR_list = []
    zce_Halton_list = []
    zce_Sobol_list = []
    zce_uMUL_list = []
    zce_DUS_list = []

    # 针对当前 lengthN，用 A/B 各自的 LFSR 序列生成阈值
    lfsr_A_thresholds = get_lfsr_thresholds_for_N(lengthN, MASTER_LFSR_A_SEQ)
    lfsr_B_thresholds = get_lfsr_thresholds_for_N(lengthN, MASTER_LFSR_B_SEQ)

    for _ in range(simulations):
        #### Halton sequence ####
        halton_engine = Halton(d=2, scramble=False)
        halton_samples_all = halton_engine.random(n=lengthN)
        halton_samples_all2 = halton_engine.random(n=lengthN)
        H1 = (halton_samples_all[:, 0] * lengthN).astype(int).tolist()
        H2 = (halton_samples_all2[:, 1] * lengthN).astype(int).tolist()

        #### Sobol sequence ####
        sobol_engine = Sobol(d=2, scramble=False)
        sobol_samples = sobol_engine.random(n=lengthN)
        sobol_samples2 = sobol_engine.random(n=lengthN)
        S1 = (sobol_samples[:, 0] * (lengthN)).astype(int).tolist()
        S2 = (sobol_samples2[:, 1] * (lengthN)).astype(int).tolist()

        #### Binary inputs ####
        binary_numberA = generate_random_8bit()
        binary_numberB = generate_random_8bit()
        decimal_numberA = binary_to_decimal(binary_numberA)
        decimal_numberB = binary_to_decimal(binary_numberB)
        NumberA = decimal_numberA / 255
        NumberB = decimal_numberB / 255
        NumberAA.append(NumberA)
        NumberBB.append(NumberB)

        ###### A streams ######
        StreamA_random = binary_to_random_bitstream(binary_numberA, lengthN)


        streamA_LFSR = generate_255bit_stream_LFSR(
            binary_numberA, lfsr_A_thresholds, lengthN
        )

        streamA_H1 = generate_255bit_stream_N3(binary_numberA, H1, lengthN)
        streamA_S1 = generate_255bit_stream_N3(binary_numberA, S1, lengthN)
        streamA_ADUS = generate_255bit_stream_N3(binary_numberA, ADUS, lengthN)

        ###### B streams ######
        StreamB_random = binary_to_random_bitstream(binary_numberB, lengthN)


        streamB_LFSR = generate_255bit_stream_LFSR(
            binary_numberB, lfsr_B_thresholds, lengthN
        )

        streamB_H2 = generate_255bit_stream_N3(binary_numberB, H2, lengthN)
        streamB_S2 = generate_255bit_stream_N3(binary_numberB, S2, lengthN)
        streamB_P2 = generate_255bit_stream_N3(binary_numberB, proposed, lengthN)
        stream_B_UMUL = generate_255bit_stream_N3_UGMEE_sequentialA(
            binary_numberB, S2, streamA_S1, lengthN
        )

        ###### Reference ######
        ref_val = NumberA * NumberB
        ref_number.append(ref_val)

        ###### MUL Errors ######
        error_MUL_Random.append(
            compute_stream_mul_error(StreamA_random, StreamB_random, ref_val, lengthN)
        )
        error_MUL_LFSR1_LFSR2.append(
            compute_stream_mul_error(streamA_LFSR, streamB_LFSR, ref_val, lengthN)
        )
        error_MUL_H1_H2.append(
            compute_stream_mul_error(streamA_H1, streamB_H2, ref_val, lengthN)
        )
        error_MUL_S1_S2.append(
            compute_stream_mul_error(streamA_S1, streamB_S2, ref_val, lengthN)
        )
        error_MUL_uMUL.append(
            compute_stream_mul_error(streamA_S1, stream_B_UMUL, ref_val, lengthN)
        )
        error_MUL_N2_S2.append(
            compute_stream_mul_error(streamA_ADUS, streamB_P2, ref_val, lengthN)
        )

       

        scc_Random_list.append(compute_scc(StreamA_random, StreamB_random))
        scc_LFSR_list.append(compute_scc(streamA_LFSR, streamB_LFSR))
        scc_Halton_list.append(compute_scc(streamA_H1, streamB_H2))
        scc_Sobol_list.append(compute_scc(streamA_S1, streamB_S2))
        scc_uMUL_list.append(compute_scc(streamA_S1, stream_B_UMUL))
        scc_DUS_list.append(compute_scc(streamA_ADUS, streamB_P2))

        zce_Random_list.append(compute_zce(StreamA_random, StreamB_random))
        zce_LFSR_list.append(compute_zce(streamA_LFSR, streamB_LFSR))
        zce_Halton_list.append(compute_zce(streamA_H1, streamB_H2))
        zce_Sobol_list.append(compute_zce(streamA_S1, streamB_S2))
        zce_uMUL_list.append(compute_zce(streamA_S1, stream_B_UMUL))
        zce_DUS_list.append(compute_zce(streamA_ADUS, streamB_P2))

    return {
        'NumberAA': NumberAA,
        'NumberBB': NumberBB,
        'ref_number': ref_number,
        'MUL_errors': {
            'Random': error_MUL_Random,
            'LFSR': error_MUL_LFSR1_LFSR2,
            'Halton': error_MUL_H1_H2,
            'Sobol': error_MUL_S1_S2,
            'uMUL': error_MUL_uMUL,
            'DUS': error_MUL_N2_S2
        },
        
        'SCC': {
            'Random': scc_Random_list,
            'LFSR': scc_LFSR_list,
            'Halton': scc_Halton_list,
            'Sobol': scc_Sobol_list,
            'uMUL': scc_uMUL_list,
            'DUS': scc_DUS_list,
        },
        'ZCE': {
            'Random': zce_Random_list,
            'LFSR': zce_LFSR_list,
            'Halton': zce_Halton_list,
            'Sobol': zce_Sobol_list,
            'uMUL': zce_uMUL_list,
            'DUS': zce_DUS_list,
        }
    }


# =============================================
# Sweep  bitstream length
# =============================================

simulations = 1000
lengthN_list = [16, 32, 64, 128, 256, 512, 1024]

methods = ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]

table_MAE = {m: [] for m in methods}
table_SCC = {m: [] for m in methods}
table_ZCE = {m: [] for m in methods}

for lengthN in lengthN_list:
    print(f"\n[ Running N = {lengthN} ]")

    #  DUS parameter
    a_dict = {16: 7, 32: 19, 64: 41, 128: 83, 256: 157, 512: 323, 1024: 629}
    a = a_dict[lengthN]

    ADUS = list(range(lengthN))

    def generate_deterministic_uniform_sequence(N, a):
        from math import gcd
        assert gcd(a, N) == 1
        return [((a * i) % N) for i in range(N)]

    proposed = generate_deterministic_uniform_sequence(lengthN, a)

    # 执行仿真
    results = simulate_and_collect_errors(simulations=simulations, lengthN=lengthN)

    # -------- MAE --------
    table_MAE["Random"].append(np.mean(results["MUL_errors"]["Random"]))
    table_MAE["LFSR"].append(np.mean(results["MUL_errors"]["LFSR"]))
    table_MAE["Halton"].append(np.mean(results["MUL_errors"]["Halton"]))
    table_MAE["Sobol"].append(np.mean(results["MUL_errors"]["Sobol"]))
    table_MAE["uMUL"].append(np.mean(results["MUL_errors"]["uMUL"]))
    table_MAE["DUS"].append(np.mean(results["MUL_errors"]["DUS"]))

    # -------- SCC --------
    table_SCC["Random"].append(np.mean(results["SCC"]["Random"]))
    table_SCC["LFSR"].append(np.mean(results["SCC"]["LFSR"]))
    table_SCC["Halton"].append(np.mean(results["SCC"]["Halton"]))
    table_SCC["Sobol"].append(np.mean(results["SCC"]["Sobol"]))
    table_SCC["uMUL"].append(np.mean(results["SCC"]["uMUL"]))
    table_SCC["DUS"].append(np.mean(results["SCC"]["DUS"]))

    # -------- ZCE --------
    table_ZCE["Random"].append(np.mean(results["ZCE"]["Random"]))
    table_ZCE["LFSR"].append(np.mean(results["ZCE"]["LFSR"]))
    table_ZCE["Halton"].append(np.mean(results["ZCE"]["Halton"]))
    table_ZCE["Sobol"].append(np.mean(results["ZCE"]["Sobol"]))
    table_ZCE["uMUL"].append(np.mean(results["ZCE"]["uMUL"]))
    table_ZCE["DUS"].append(np.mean(results["ZCE"]["DUS"]))


# =============================================
#            打印三张表格 
# =============================================

def print_table(title, table):
    print("\n==============================")
    print(f"=== {title} ===")
    print("==============================")

    # 列标题
    header = "Method     | " + "  ".join([f"{N:6d}" for N in lengthN_list])
    print(header)
    print("-" * len(header))

    # 行数据
    for m in methods:
        row_values = "  ".join([f"{v:.6f}" for v in table[m]])
        print(f"{m:<10} | {row_values}")


print_table("MUL MAE Table", table_MAE)
print_table("SCC Table", table_SCC)
print_table("ZCE Table", table_ZCE)

