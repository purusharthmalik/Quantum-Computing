from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from PIL import Image
from qiskit.quantum_info import Statevector

# Quantum encoding methods
def quantum_encode_amplitude(pixel_value):
    normalized_pixel = pixel_value / 255.0
    qc = QuantumCircuit(1)
    theta = 2 * np.arcsin(np.sqrt(normalized_pixel))
    qc.ry(theta, 0)
    return qc

def quantum_encode_phase(pixel_value):
    normalized_pixel = pixel_value / 255.0
    qc = QuantumCircuit(1)
    qc.h(0)
    phi = normalized_pixel * np.pi
    qc.p(phi, 0)
    return qc

def quantum_encode_basis(pixel_value):
    basis_state = int(np.round(pixel_value / 255.0))
    qc = QuantumCircuit(1)
    if basis_state == 1:
        qc.x(0)
    return qc

def display_bloch_sphere(statevector):
    return plot_bloch_multivector(statevector, title="Quantum State Representation")

def analyze_quantum_state(statevector):
    # Probability amplitudes
    prob_zero = np.abs(statevector[0])**2
    prob_one = np.abs(statevector[1])**2
    
    # Bloch sphere coordinates
    bloch_coords = {
        'x': 2 * np.real(statevector[0] * np.conj(statevector[1])),
        'y': 2j * np.imag(statevector[0] * np.conj(statevector[1])),
        'z': prob_one - prob_zero
    }
    
    return {
        'Probability |0⟩': prob_zero,
        'Probability |1⟩': prob_one,
        'Bloch Coordinates': bloch_coords
    }

def quantum_encode_image(image_array, encoding_method):
    encoding_heatmap = np.zeros_like(image_array, dtype=float)
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            if encoding_method == 'Amplitude Encoding':
                qc = quantum_encode_amplitude(image_array[y, x])
                statevector = Statevector.from_instruction(qc)

                # Using probability of |1⟩ state as heatmap value
                encoding_heatmap[y, x] = np.abs(statevector[1])**2
            elif encoding_method == 'Phase Encoding':
                qc = quantum_encode_phase(image_array[y, x])
                statevector = Statevector.from_instruction(qc)

                # Using the phase as heatmap value
                encoding_heatmap[y, x] = np.arctan(statevector[1]/statevector[0])
            else:
                qc = quantum_encode_basis(image_array[y, x])
                statevector = Statevector.from_instruction(qc)

                # Heatmap values in a binary fashion
                encoding_heatmap[y, x] = 0 if statevector[0] != 0 else 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(encoding_heatmap, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Quantum Encoding Intensity')
    st.pyplot(fig)

def main():
    st.title("Quantum Encoding and Visualization")
    st.sidebar.header("🔬 Quantum Encoding Techniques")
    st.sidebar.markdown("""
    ### How Quantum Encoding Works
    
    Quantum encoding transforms classical data into quantum states, enabling unique computational capabilities:
    
    #### 1. Amplitude Encoding 📊
    - Maps pixel intensity to quantum state amplitudes
    - Preserves proportional information
    - Rotation around Y-axis (Ry gate)
    - Best for representing continuous values
    
    #### 2. Phase Encoding 🌀
    - Maps pixel intensity to quantum phase
    - Encodes information in phase angle
    - Rotation around Z-axis (Rz gate)
    - Useful for preserving phase relationships
    
    #### 3. Basis Encoding 🔢
    - Maps pixel to binary quantum state (|0⟩ or |1⟩)
    - Simple binary representation
    - Uses X-gate to flip state
    - Ideal for binary or threshold-based data
    """)

    uploaded_file = st.file_uploader("Upload a grayscale image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        image_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Selecting a pixel to encode
        st.subheader("Select Pixel for Quantum Encoding")
        x = st.slider("X-coordinate", 0, image_array.shape[1] - 1, 0)
        y = st.slider("Y-coordinate", 0, image_array.shape[0] - 1, 0)
        pixel_value = image_array[y, x]

        st.write(f"Selected Pixel Value: {pixel_value}")

        encoding_method = st.selectbox(
            "Choose Quantum Encoding Method",
            ["Amplitude Encoding", "Phase Encoding", "Basis Encoding"]
        )

        if encoding_method == "Amplitude Encoding":
            quantum_circuit = quantum_encode_amplitude(pixel_value)
        elif encoding_method == "Phase Encoding":
            quantum_circuit = quantum_encode_phase(pixel_value)
        else:
            quantum_circuit = quantum_encode_basis(pixel_value)

        statevector = Statevector.from_instruction(quantum_circuit)

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(display_bloch_sphere(statevector), use_container_width=True)

        with col2:
            st.pyplot(quantum_circuit.draw(output='mpl'), use_container_width=False)

        quantum_state_analysis = analyze_quantum_state(statevector)
        st.subheader("Quantum State Analysis")
        st.json(quantum_state_analysis)

        st.subheader("Image-wide Quantum Encoding")
        quantum_encode_image(image_array, encoding_method)

if __name__ == "__main__":
    main()