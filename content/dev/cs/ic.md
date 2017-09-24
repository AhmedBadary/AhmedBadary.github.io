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
  * [SECOND](#content2)
  {: .TOC2}
  * [THIRD](#content3)
  {: .TOC3}
  * [FOURTH](#content4)
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


2. **Symmetric Key Encryption:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} 
    :   When both the sender and receiver have the same key to encrypt and decrypt.

3. **Eve's Capabilities:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13} \\
    * **NO knowledge of K:**
    * **Recognition of Success:**


4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents14} 


5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents15} 


6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents16} 


7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents17} 


8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents18} 


***

## n-Time Pads
{: #content2}

<p class="message"> Provably secure crypto-systems</p>

1. **One-Time Pad:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21} \\
    * We use a different key for each message M.  
        * Different = Completely Independent.
    * Make the key as long as M.
    *$$E(M,k) = M \xor K$$.  
    * **Proof of security:**  
        Since K is random and independent, all possible bit-patterns for C are equally likely and the attacker cannot know anything about the system even from random text.  
        Thus, observing a given C does not help Eve narrow down the possibilities whether Alice chose M' or M''.  

    * **Issues:**  
        1. **Key-Generation:** Need truly Random, Indepndent keys. Those are very very difficult to generate.
        2. Key-Distribution:** Need to share keys as long as all possible communication

2. **Two-Time Pads:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22} \\
    * Reusing a key K one more time.
    * Alice sends C = E(M,K) and C'=E(M',K).  
    * Eve Obseves $$M \xor K \text{ and } M' \xor K$$ and computes $$C \xor C' = (M \xor K) \xor (M' \xor K) = M \xor M'$$.  
    * Now, Eve knows everything about M'.
    * Thus, it is insecure.


***

## Modern Encryption
{: #content3}

1. **Block Cypher:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31} \\


2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} \\

