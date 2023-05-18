# Operations for Constructing QECC Tensor Network

Operations allowed for constructing a QECC tensor network $T$.
$$
\begin{aligned}
T::= & ~ \textbf{const}~C \mid \textbf{contract}~T_a~[a_1,...,a_n]~ T_b  ~[b_1,...,b_n] \\
&\mid \textbf{self-con}~ T'~[a_1,...,a_n]~[b_1,...,b_n]\mid \textbf{set}~T'~i~[log, phy] 
\end{aligned}
$$

- $T::=\textbf{const}~C$ means $T$ is constructed by a known tensor $C$
- $T::=\textbf{contract}~T_a~[a_1,...,a_n]~ T_b~[b_1,...,b_n]$ means $T$ is constructed by contracting the legs indexed by $[a_1,...,a_n]$ in a tensor network $T_a$ with the legs indexed by $[b_1,...,b_n]$ in a tensor network $T_b$ (i.e. contract leg $a_i$ with $b_i$, $i=1,2,...,n$). 
- $T::=\textbf{self-con}~ T'~[a_1,...,a_n]~[b_1,...,b_n]$ means $T$ is constructed by self-contracting the legs indexed by $[a_1,...,a_n]$ and $[b_1,...,b_n]$ in $T'$ repsectively.
- $T::=\textbf{set}~T'~i~[\text{log} \mid \text{phy}]$ means to set the leg $i$ in $T'$ to a logical leg or a physical leg.

We may also need to construct a QECC tensor network with a for loop structure

$$
T::= \textbf{loop}~T'~m~[a_1,...,a_i]~[b_1,...,b_i]
$$
In this structure, we restrict that the index number $a_1,...,a_i,b_1,...,b_i$ are different from each other.
This structure means to construct $T$ with $m$ tensor networks $T'$ by recusively constracting the leg indexed by $[a_1,...,a_i]~[b_1,...,b_i]$. For example, suppose $T'$ is a tensor network with four legs indexed by $0,1,2,3$
```
     0
     |
3 -- T' -- 1
     |
     2
```
$\textbf{loop}~T'~3~[0]~[2]$ means to contruct a tensork network with three $T'$ ($T'_1,T'_2,T'_3$), by contracting leg 2 of $T'_1$ with leg 0 of $T'2$ and contracting leg 2 of $T'2$ with leg 0 of $T'_3$
```
     0
     |
3 -- T'1 -- 1
     |
     2
     |
     0
     |
3 -- T'2 -- 1
     |
     2
     |   
     0
     |
3 -- T'3 -- 1
     |
     2  
```

# Quantum weight enumerators

Given a $[[n,k]]$ QECC $C$, we can use the term $|A_C(p)-B_C(p)|$ expresses the probability of an undetected error under unbiased depolarizing noise model. 
$$
\begin{aligned}
&A_C(p) = 4^k\cdot \sum_{d=0}^n |P^n[d] \cap S |\cdot p^d\\
&B_C(p) = 2^k\cdot \sum_{d=0}^n |P^n[d] \cap N |\cdot p^d
\end{aligned}
$$
Here $P^n$ is the Pauli group, $P^n[d]$ contains the $d$-weight elements in $P^n$. $S$ is the stabilizer group of $C$ and $N$ is the normalizer group of $C$. Basically the intuition is that A and B are counting the number of stabilizer and normalizers. For example, the $[[5,1,3]]$ code:
$$
\begin{aligned}
&\frac{1}{4}\cdot A_{[[5,1,3]]}(p) = 1 + 15p^4\\
&\frac{1}{2}\cdot B_{[[5,1,3]]}(p) = 1 + 30p^3+ 15p^4 + 18p^5
\end{aligned}
$$

If we are using the biased depolarizing noise model, for each weight-$d$ operator in the Pauli group, suppose $d=d_x+d_y+d_z$, change the term $p^d$ to
$$
p_x^{d_x}p_y^{d_y}p_z^{d_z}
$$

<!-- How do we get $z$ from a noise model? For example, for the depolarizing noise model, can we use the error rate $p$ as $z$ to calculate this cost function? If we want a biased error rate for Pauli X Y Z operators (i.e. $p_x\neq p_y \neq p_z$), how should we calculate these $|A_C(z)-B_C(z)|$? -->

   
<!-- 2. I cannot fully understand how to compute $A_C(z), B_C(z)$. For example, in the example III.1 in the paper, we have 
$$
\begin{aligned}
&\frac{1}{4}\cdot A_{[[5,1,3]]}(z) = 1 + 15z^4\\
&\frac{1}{2}\cdot B_{[[5,1,3]]}(z) = 1 + 30z^3+ 15z^4 + 18z^5
\end{aligned}
$$
I can see that they are computed from the equations
![](Aeq.JPG)
![](Beq.JPG) -->