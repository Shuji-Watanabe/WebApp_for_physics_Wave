import numpy as np
import matplotlib.pyplot as plt
import streamlit as st





# --- 関数定義 ---
def compute_acceleration(position, velocity, k, m, c):
    """加速度を計算する関数"""
    return -(k / m) * position - (c / m) * velocity

def run_simulation(initial_position, initial_velocity, k, m, c, t_max, dt):
    """シミュレーションを実行する関数"""
    num_steps = int(t_max / dt) + 1
    time = np.linspace(0, t_max, num_steps)
    position = np.zeros(num_steps)
    velocity = np.zeros(num_steps)

    # 初期条件の設定
    position[0] = initial_position
    velocity[0] = initial_velocity

    # 数値積分で運動方程式を解く
    for i in range(1, num_steps):
        acceleration = compute_acceleration(position[i - 1], velocity[i - 1], k, m, c)
        velocity[i] = velocity[i - 1] + acceleration * dt
        position[i] = position[i - 1] + velocity[i - 1] * dt

    return time, position, velocity

def plot_results(time, position, velocity):
    """シミュレーション結果をプロットする関数"""
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # 位置のプロット
    ax[0].plot(time, position, label="Position", color="blue")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    ax[0].grid()

    # 速度のプロット
    ax[1].plot(time, velocity, label="Velocity", color="orange")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend()
    ax[1].grid()

    st.pyplot(fig)

# --- Streamlit UI ---
st.header("波動の数理支援用アプリ",divider="rainbow")
st.write("このアプリでは、バネ-質量-減衰器システムの運動をシミュレーションします。")

# パラメータ入力
st.sidebar.header("シミュレーションパラメータ")
initial_position = st.sidebar.number_input("初期位置 (m):", value=1.0)
initial_velocity = st.sidebar.number_input("初期速度 (m/s):", value=0.0)
k = st.sidebar.number_input("バネ定数 k (N/m):", value=10.0)
m = st.sidebar.number_input("質量 m (kg):", value=1.0)
c = st.sidebar.number_input("減衰係数 c (Ns/m):", value=0.5)
t_max = st.sidebar.number_input("シミュレーション時間 (s):", value=10.0)
dt = st.sidebar.number_input("時間刻み幅 dt (s):", value=0.01)

# シミュレーション実行ボタン
if st.sidebar.button("シミュレーション開始"):
    # シミュレーションの実行
    time, position, velocity = run_simulation(initial_position, initial_velocity, k, m, c, t_max, dt)

    # 結果のプロット
    plot_results(time, position, velocity)

    # 数値結果の表示
    st.write("### 数値結果")
    st.write("位置と速度の最初の10ステップを表示します:")
    result_table = np.vstack((time, position, velocity)).T[:10]
    st.write("""| 時間 (s) | 位置 (m) | 速度 (m/s) |
|:---------:|:--------:|:---------:|""")
    for row in result_table:
        st.write(f"| {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} |")