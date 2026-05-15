 Quantum Error Correction Using Machine Learning vs Repetition Decoding
Overview
----------------------------------------------------------------------------
This project explores how machine learning can improve quantum error correction compared to classical repetition decoding in noisy multi-qubit systems. A simulated quantum dataset is generated with controlled noise, and two decoding approaches are compared:

Classical repetition decoding (majority vote)
Machine learning decoder (Gradient Boosting classifier)
------------------------------------------------------------
 Question

Can a machine learning model outperform classical repetition decoding in correcting errors in noisy multi-qubit quantum systems?

Hypothesis
-------------------------------------------------------

Machine learning will outperform repetition decoding because it can learn non-uniform and correlated noise patterns across qubits, while repetition decoding treats all qubits equally.

Methodology
----------------------
1. Data Generation
5-qubit system 
Logical states encoded as:
0 → 00000
1 → 11111
Noise is applied with:
different and random error rates with each  qubit
correlated bit flips
rare burst errors
2. Decoding Methods
 Repetition Decoder
Uses majority vote across qubits
Assumes all qubits are equally reliable
*
   Machine Learning Decoder
Learns patterns in noise qubit data
Identifies qubit reliability difference

 Results
--------------------------
Method	Accuracy
Repetition Decoder	~0.738
Machine Learning Decoder	~0.800

Key Insight
-----------------------

Machine learning improves performance by learning that some qubits (e.g., q0 and q4) are more reliable than others under noisy conditions.

Key Findings
------------------------------
Qubit reliability is not uniform
ML identifies important qubits automatically
Repetition decoding is limited by majority voting
ML adapts to correlated and structured noise

Technologies Used
-------------
Python
Pandas
NumPy
Scikit-learn
Matplotlib



Future Work
-------------------
Test on larger qubit systems
Apply neural networks
Use real quantum hardware noise models
Explore hybrid quantum-classical decoders



Thank You!
