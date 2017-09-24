---
layout: NotesPage
title: Introduction <br /> Symmetric-Key Cryptography
permalink: /work_files/dev/cs/ic
prevLink: /work_files/dev/cs.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [What is Cryptography?](#content1)
  {: .TOC1}
  * [Caeser Ciphers](#content2)
  {: .TOC2}
  * [n-time Pads](#content3)
  {: .TOC3}
  * [Modern Block Cyphers](#content4)
  {: .TOC4}
  * [FIFTH](#content5)
  {: .TOC5}
  * [SIXTH](#content6)
  {: .TOC6}
</div>

***
***

## What is Cryptography?
{: #content1}

1. **Three Main Goals:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11} \\
    1. **Confidentiality:** Preventing adversiries from reading our data.
    2. **Integrity:** Preventing attackers from altering our data. (regardless of its privacy)
    3. **Authentication:** Proving who created a given message or document.


11. **The Set Up:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents111}  
    :   * **Who's involved:**  
            1. _Alice:_ Sender. 
            2. _Bob:_ Receiver.
            3. _Eve:_ Eavesdropper.
            4. _Mallory:_ Manipulator.
    :   * **Goals:**  
            1. _Attackers Goal:_ Any Knowledge on $$M_i$$ besides a bound on it's length.
            2. _Defenders Goal:_ Ensure attakcer has no reason to think any $$M' \in \{0,1\}^n$$ is more likely than any other.  
            > Where $$len(M_i) = n$$.
    :   * **Current Situation:** 
            1. _Eve's Capabilities:_ 
                * No knowledge of the key $$K$$.
                * **Recognition of Success**.  
                > Eve can tell if she fully (NOT PARTIALLY) recovered $$M_i$$.  

            2. _The key $$(k)$$_:  
                * Assume $$k$$ is selected completely randomly.
                * For a $$b$$-bit key, any $$k \in \{0,1\}^b$$ is equally likely.

111. **Terminology:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents1111}  \\
    1. **Plaintext $$(M_i)$$:** The message to be sent.   
    2. **Cyphertext $$(C_i)$$:** The encrypted message that is sent.  
    3. **Key $$(K)$$:**   
    4. **Encryption Function $$(E(M_i, K))$$:** A function of the message and the key that outputs the _cyphertext_.  
    5. **Decryption Function $$(D(C_i,K))$$:** 


2. **Types of Attacks:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} \\
    1. **Cyphertext-only Attack:**
        * Eve sees every instance of $$C_i$$.
        * Eve knows some partial information (Variant).  
        > e.g. message is in English.
    2. **Known Plaintext:**  
        * Eve knows part of $$M_i$$ and/or entire other $$M_j$$'s.
    3. **Chosen Plaintext:** 
        * Eve gets Alice to send $$M_j$$'s of Eve's choosing.  
    4. **Chose Cyphertext:**  
        * Eve tricks Bob into decrypting some $$C_j$$ of her choice and he reveals something about the result.
    5. **Combination of all above:**
        * Idealy, we would like to defend against this.

22. **Symmetric Key Encryption:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents122} 
    :   When both the sender and receiver have the same key to encrypt and decrypt.

3. **Independence Under Chosen Plaintext Attack game (IND=CPA):**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13} \\
    * **Eve is interacting with an encryption "Oracle":** The Oracle has an unknown key $$k$$.
    * **Eve can provide TWO seperate chosen plaintexts of the same length:** Oracle will randomly select one to encrypt with the unknown key; the game can _repeat_.
    * **Eve's Goal:** To have a better thatn random chance of guessing which plaintext the Oracle selected.  
        > Variations include the Orcale always selecting the first or the second.

    * **To be Independent under Chose Plaintext:** Eve must not be able to achieve her goal.

4. **Kerckhoffs Principle:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents14} \\
    1. Cryptosystems should remain secure even when the attakcer knows all the internal details.  
        > Don't rely on security by 0bs$$\in$$cur!ty.
    2. Key should be the only thing that must stay secret.
    3. It should be easy to change keys.

***

## Caeser Ciphers
{: #content2}

1. **ROTK and ROT-K:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21}
    :   **ROTK** is an encryption function $$(E(M,K)=ROTK(M))$$ that takes each letter in $$M$$ and 'rotates' it $K$$ positions (with-wrapping) through the alphabet.  
    :   **ROT-K** is a decryption function $$(D(C,K)=ROT-K(C))$$ that takes each letter in $$M$$ and 'rotates' it $26-K$$ positions (with-wrapping) through the alphabet.  

2. **Attacks on Caesar Ciphers:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22} \\
    1. **Brute-Force:** try every possible value of k.
    2. **Deduction:** (_known plaintext_) Analyze letter frequencies - (_chosen plaintext_) Guess possible words.

3. **Enigma:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23} 
    Uses Caesar Ciphers but with a new key for each letter.

## n-time Pads

0. **The XOR $$(\oplus)$$ function:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents30}
    :    $$
        \\
        \begin{align}
        X \oplus 0 &= X \\
        X \oplus X &= 0 \\
        X \oplus Y &= Y \oplus X \\
        X \oplus (Y \oplus Z) &= (X \oplus Y) \oplus Z
        \end{align}
        $$


1. **One-Time Pad:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31} \\
    * We use a different key for each message M.  
        * Different = Completely Independent.
    * Make the key as long as M.
    * **$$E(M,k) = M \oplus K$$**.  
    * **Proof of security:**  
        Since K is random and independent, all possible bit-patterns for C are equally likely and the attacker cannot know anything about the system even from random text.  
        Thus, observing a given C does not help Eve narrow down the possibilities whether Alice chose M' or M''.  

    * **Issues:**  
        1. **Key-Generation:** Need truly Random, Indepndent keys. Those are very very difficult to generate.
        2. **Key-Distribution:** Need to share keys as long as all possible communication.

2. **Two-Time Pads:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} \\
    * Reusing a key K one more time.
    * Alice sends C = E(M,K) and C'=E(M',K).  
    * Eve Obseves $$M \oplus K \text{ and } M' \oplus K$$ and computes $$C \oplus C' = (M \oplus K) \oplus (M' \oplus K) = M \oplus M'$$.  
    * Now, Eve knows everything about M'.
    * Thus, it is insecure.


***

## Modern Encryption
{: #content3}

1. **Block Cypher:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31} \\
    * A function $$E : \{0, 1\}^b \times \{0, 1\}^k \rightarrow \{0, 1\}^b$$.
    * [For a fixed $$K$$] $$E_K : \{0, 1\}^b \rightarrow \{0, 1\}^b$$,  denoted by $$E_K(M) = E(M,K)$$.
    * **Properties:**  
        1. _Correctness:_  $$E_K(M)$$ is a bijective permutation function.
        2. _Efficiency:_ computable in microseconds.
        3. _Security:_ For unknown $$K$$, "behaves" like a random permutation.

2. **Data Encryption Standard (DES):**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} \\
    * Designed in the Seventies.
    * Block Size 64 bits, Key size 56 bits.
    * Tweaked by the NSA.

3. **Advanced Encryption Standard (AES):**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} \\
    * 20 years old, Standarized 15 years ago.
    * Block size 128 bits.
    * Key size 128, 192, 256 bits.
    * Assumed (but not proven) secure. 

4. **Analysis on Brute-Forcing:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} \\
    * $$2^128 \leq 10^39$$ possibilities.
    * CPU capacity $$10^9 = 1b$$ keys/second $$ = 10^18$$ keys/sec.
    * Thus, we will need $$10^21$$ seconds, or approximately, $$3*10^13 = 30$$ trillion years.

5. **Drawbacks of Block Ciphers:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents35} \\
    * Block Ciphers can only encrypt messages of a certain size.
        * If $$M$$ is smaller, then just pad it.
        * If $$M$$ is larger, then repeatedly apply block cipher.


## Modern Block Ciphers

1. **Electronic Code Book Mode(ECB):**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41} \\
    * Simplest Block Cipher mode.
    * Split the message into b-bit blocks $$P_1, P_2, \cdots$$.
    * Each block is enciphered independently, separate from the other blocks $$C_i = E(P_i, K)$$.
    * Since the key $$K$$ is fixed, each block is subject to the same permutation.

    ![img](/main_files/cs/ic/all.png){: width="100%"}

    * **Issues:**
        * Not IND-CPA
        * Thus, the relationship between the plaintext is reflected in the cyphertext.

2. **Building a better Cipher:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} \\
    1. Ensure blocks incorporate more than just the plaintext to mask relationships between blocks:
        1. Include elements of prior computation.
        2. (OR) Include positional information.
    2. Add initial Randomness: 
        1. Prevent encryption scheme from determinism revealing relationships between messages.
        2. Introduce initialization vector (IV).  
            > IV is a public nonce, a use-once unique value: Easiest way to get one is generate it randomly

3. **CBC Encryption:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43}
    :

    :   * **Algorithm (Encryption)**  
            * If b is the block size of the block cipher, split the plaintext in blocks of size $$b: P_1, P_2, P_3, \cdots$$.
            * Choose a random IV (do not reuse for other messages)
            * Define $$C_0 = IV$$.
            * The ith ciphertext block is given by $$C_i = E_K(C_{i−1} \oplus M_i)$$.
            * The ciphertext is the concatenation of the initial vector and these individual blocks: $$C = IV \cdot C_1 \cdot C_2 \cdots C_l$$.
    :   * **Algorithm (Decryption)**  
            * Take the $$IV$$ out of the ciphertext.
            * If $$b$$ is the block size of the block cipher, split the ciphertext in blocks of size $$b: C_1, C_2, C_3, \cdots$$.
            * The ith plaintext block is given by $$P_i = D_K(C_{i−1} \oplus M_i)$$.
            * Output the plaintext as the concatenation of $$P_1, P_2, P_3, \cdots$$.
    :   * **Properties:**   
            * Widely used.  
            * _Security:_ If no reuse of nonce, both are provably secure (IND-CPA) assuming the underlying block cipher is secure.
            * REusing the $$IV$$ voids IND-SPA but is not terrible.
    :   * **Issues:**
            * Sequential encryption; we can't parallelize encryption.
            * Must finish encrypting block $$b$$ before starting $$b+1$$.  
            > But you can parallelize decryption
    :   ![img](/main_files/cs/ic/all2.png){: width="100%"}

4. **CTR Mode:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents44}
    :   We avoid the issue of not non-parallizable ciphers by by encrypting a counter initialized to IV to obtain a sequence that can now be used as though they were the keys for a one-time pad:  
    namely, $$Z_i = E_K(IV + i) \ $$ and $$\ C_i = Z_i \oplus M_i$$.  
        > Uses the encryptor for decrypting.  
        > CTR mode is a _Stream Cipher_.
    :   REusing the $$IV \implies$$ no more security.
    :   ![img](/main_files/cs/ic/all3.png){: width="100%"}

5. **CFB Mode:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents45}
    :   
    :   * **Properties:**  
            * Since the encryption is XORed with the plaintext.
                > It is more convineant than CBC since it can end on a "short" block without a problem.
            * Similar security properties as CBC mode: 
                1. Sequential encryption, parallel decryption.
                2. Same error propagation effects.
                3. Effectively the same for IND-CPA.
            * Worse than CBC if you reuse the $$IV$$.
    :   ![img](/main_files/cs/ic/all4.png){: width="100%"}
