---
layout: NotesPage
title: Memory Safety <br /> Vulnerabilities and Prevention 
permalink: /work_files/dev/cs/ms
prevLink: /work_files/dev/cs.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Buffer Overflow Vulnerabilities](#content1)
  {: .TOC1}
  * [Stack smashing](#content2)
  {: .TOC2}
  * [Format String Vulnerabilities](#content3)
  {: .TOC3}
  * [Integer Conversion Vulnerabilities](#content4)
  {: .TOC4}
  * [Other Vulnerabilities](#content5)
  {: .TOC5}
  * [Defending against Memory-Vulnerabilities](#content6)
  {: .TOC6}
</div>

***
***

## **Memory Safety:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
<p class="message">
Ensuring the integrity of a programs data structures by preventing users (attackers) from reading and writing to memory at will (location- and time- wise).
</p>

## Buffer Overflow Vulnerabilities
{: #content1}

1. **What?**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
    :   A buffer overflow bug is one where the programmer fails
to perform adequate bounds checks, triggering an out-of-bounds memory access that writes
beyond the bounds of some memory region.

2. **How?**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12}
    :   Attackers can use these out-of-bounds memory accesses to corrupt the program’s intended behavior.

3. **Why? (Who's at risk?)**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13}
    :   This is a common problem with the language C. 
    :   C, has no "automatic Bounds Checking" for array or pointer access.

4. **Example:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents14}
    1. In this example, gets() reads as many bytes of input as are available on standard input, and stores them into buf[]. If the input contains more than 80 bytes of data, then gets() will write past the end of buf, overwriting some other part of memory.  
    ```c
        char buf[80];
        void vulnerable() {
            gets(buf);
        }
    ```
    > This is a bug. This bug typically causes a crash and a core-dump.

    2. Unfortunately, the authenticated flag is stored in memory right after ```buf```. If the attacker can write 81 bytes of data to buf (with the 81st byte set to a non-zero value), then this will set the authenticated flag to true, and the attacker will gain access.  
    ```c
        char buf[80];
        int authenticated = 0;
        void vulnerable() {
            gets(buf);
        }
    ```
    > An attacker who can control the input to the program can bypass the password checks.

    3. The function pointer ```fnptr``` is invoked elsewhere in the program (not shown). This enables a more serious attack: the attacker can overwrite fnptr with any address of their choosing, redirecting program execution to some other memory location.  
    ```c
        char buf[80];
        int (*fnptr)();
        void vulnerable() {
            gets(buf);
        }
    ```
    > A crafty attacker could supply an input that consists of malicious machine instructions, followed by a few bytes that overwrite fnptr with some address A.
    When fnptr is next invoked, the flow of control is redirected to address A.   
    > > Of course, many variations on this attack are possible:  
    > > for instance, the attacker could arrange to store the malicious code anywhere else  e.g., in some other input buffer), rather than in buf, and redirect execution to that other location.

5. **Comments on Buffer Overflows as a Secuirty Vulnerability:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents15}
    :   Buffer overflow vulnerabilities and malicious code injection are a favorite method used by worm writers and attackers.

6. **Mitigation:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents16}
    1. Is it possible to prevent the attacker from overwriting the return address on the stack to change the control flow? Can we, at least, detect it?
        * **Solution:**  
            **Stack Canaries:** A canary or canary word is a known value placed between the local variables and control data on the stack. Before reading the return address, code inserted by the compiler checks the canary against the known value. Since a successful buffer overflows needs to overwrite the canary before reaching the return address, and the attacker cannot predict the canary value, the canary validation will fail and stop execution prior to the jump.
            ```c
                /* This number is randomly set before each run. */
                int MAGIC = rand();
                void vuln() {
                    int canary = MAGIC;
                    char buf[32];
                    gets(buf);
                    if (canary != MAGIC)
                        HALT();
                }
            ```
        * **Limitations:**  
            1. Canaries only protect against stack smashing attacks, not against heap overflows or format string vulnerabilities.
            2. Local variables, such as function pointers and authentication flags, can still be overwritten.
            3. No protection against buffer underflows. This can be problematic in combination with the previous point.
            4. If the attack occurs before the end of the function, the canary validation does not even take place. This happens for example when an exception handler on the stack gets invoked before the function returns.
        * **Cost:**  
            The canary has to be validated on each function return. The performance overhead is only a few percent since a canary is only needed in functions with local arrays.
    2. The overwritten return address must point to a valid instruction sequence. The attacker often places the malicious code to execute in the vulnerable buffer. However, the buffer address must be known to set up the jump target correctly. One way to find out this address is to observe the program in a debugger.  
    What could be done to make it harder to accurately find out the address of the start of the malicious code?
        * **Solution:**  
            **Address Randomization:** When the OS loader puts an executable into memory,   it maps the different sections (text, data/BSS, heap, stack) to fixed memory-safety locations. In the mitigation technique called address space layout Randomization(ASLR), rather than deterministically allocating the process layout, the OS randomizes the starting base of each section.
            > For instance, the OS might decide to start stack frames from somewhere other than the highest memory address.

        * **Limitations:**  
            1. Entropy reduction attacks can significantly lower the efficacy of ASLR.
            > For example, reducing factors are page alignment requirements (stack: 16 bytes, heap: 4096 bytes).
            2. Address space information disclosure techniques can force applications too leak known addresses.
            3. Revealing addresses via brute-forcing can also be an effective techniques when an application does not terminate.
            > e.g., when a block that catches exceptions exists.
            4. Techniques known as heap spraying and JIT spraying allow an attacker to inject code at predictable locations.

        * **Cost:**  
            The overhead incurred by ASLR is negligible.

    3. Attackers often store their malicious code inside the same buffer that they overflow.
    What mechanism could prevent the execution of the malicious code? What type of
    code would break with this defense in place?
        * **Solution:**  
            **Executable Space Protection:** Modern CPUs include a feature to mark certain memory regions non-executable. AMD calls this feature the NX (no execute) bit and Intel the XD (execute disable) bit. The idea is to combat buffer overflows where the attacker injects their own code.

        * **Limitations:**  
            1. An attacker does not have to inject their own code. It is also possible to leverage existing instruction sequences in memory and jump to them. See part 3 for details.
            2. The defense mechanism disallows execution of code generated at runtime, such as during JIT compilation or self-modifying code (SMC).
            3. If code is loaded at predictable addresses, it is possible to turn non-executable into executable code
            > e.g.,  via system functions like VirtualAlloc or VirtualProtect on Windows

        * **Cost:**  
            There is no measurable overhead due to the hardware support of modern CPUs.

***

## Stack smashing
{: #content2}
* [BackGround: Memory Layout, Convention, and Protocol in (Intel (x86))](/work_files/web_dev/security/ms#bodyContents24)

1. **What?**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21}
    :   When a _stack frame_ is pushed on the stack (e.g. a function stack frame), a certain amount of memory is allocated for the inputs. If the input is actually bigger than the memory allocated for it, then it will overflow and will overwrite the stack pointer (SP) and the return address.

2. **How?**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22}
    :   Stack smashing can be used for malicious code injection.
    {: style="color: black"}
    * First, the attacker arranges to infiltrate a malicious code sequence somewhere in the program’s address space, at a known address (perhaps using techniques previously mentioned). 
    * Next, the attacker provides a carefully-chosen 88-byte input, where the last four bytes hold the address of the malicious code.
    * The gets() call will overwrite the return address on the stack with the last 4 bytes of the input—in other words, with the address of the malicious code.
    * When vulnerable() returns, the CPU will retrieve the return address stored on the stack and transfer control to that address, handing control over to the attacker’s malicious code.

3. **Example:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23} \\
    1. In this example,     When vulnerable() is called, a stack frame is pushed onto the stack. The stack will look something like this:
    ![img](/main_files/web_dev/images/e2.png){: width="87%"} \\
    If the input is too long, the code will write past the end of buf and the saved SP and return address will be overwritten.  
    ```c
        void vulnerable() {
            char buf[80];
            gets(buf);
        }
    ```

4. **Memory Layout:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents24} \\
    * **The Text:** region contains the executable code of the program.
    * **The Heap:** stores dynamically allocated data (and grows and shrinks as objects are allocated and freed).
    * **The Stack:** stores local variables and other information associated with each function call (which grows and shrinks with function calls and returns).

    > Notice that the text region starts at smaller-numbered memory addresses (e.g., 0x00..0), and the stack region ends at larger-numbered memory addresses (0xFF..F).

5. **Convention and Protocol:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents25} \\
    <button>Show List</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    ![img](/main_files/web_dev/images/lst.png){: width="87%" hidden=""}


***

## Format String Vulnerabilities
{: #content3}

1. **What?**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31}
    :   A vulnerability caused by unchecking the format of the user input which can leave the system vulnerabule to crashing or injection.

2. **How?**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} 
    :   A malicious user may use the ```%s``` and ```%x``` format tokens, among others, to print data from the call stack or possibly other locations in memory. One may also write arbitrary data to arbitrary locations using the ```%n``` format token, which commands ```printf()``` and similar functions to write the number of bytes formatted to an address stored on the stack.

3. **Why? (Who's at risk?)**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} 
    :   The problem stems from the use of unchecked user input as the format string parameter in certain C functions that perform formatting, such as ```printf()```.

4. **Example:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} \\
    * In this example, If buf contains any ```%``` characters, ```printf()``` will look for non-existent arguments, and may crash or core-dump the program trying to chase missing pointers.  
    ```c
        void vulnerable() {
            char buf[80];
            if (fgets(buf, sizeof buf, stdin) == NULL)
                return;
            printf(buf);
        }
    ```

    * <button>More Examples</button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    ![formula](/main_files/web_dev/images/strng_frmt.png){: hidden="" width="70%"}

5. **Attack Types:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents35}
    * **The attacker can learn the contents of the function’s stack frame:** 
        1. Supplying the string ```"%x:%x"``` reveals the first two words of stack memory.

    * **The attacker can also learn the contents of any other part of memory, as well:** 
        1. Supplying the string ```"%s"``` treats the next word of stack memory as an address, and prints the string found at that address.  
        1. Supplying the string ```"%x:%s"``` treats the next word of stack memory as an address, the word after that as an address, and prints what is found at that string. 
        1. To read the contents of memory starting at a particular address, the attacker can find a nearby place on the stack where that address is stored, and then supply just enough ```%x```’s to walk to this place followed by a ```%s```.
        1. Thus, an attacker can exploit a format string vulnerability to learn passwords, cryptographic keys, or other secrets stored in the victim’s address space.

    * **The attacker can write any value to any address in the victim’s memory:**
        2. Use ```%n``` and many tricks; the details are beyond the scope of this writeup.  
        2. You might want to ponder how this could be used for malicious code injection.

6. **Format String:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents36}
    :   A **Format String** is an ASCIIZ string that contains text and format parameters.
    * **Example:**  ```printf(“my name is:%s\n”,”saif”);```
    > Think of a format string as a specifier which tells the program the format of the output

    * **Types:**  \\
    ![img](/main_files/web_dev/images/table.png){: width="87%"}

7. **What does the stack look like during a “printf”?**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents37}
    :   > **Command:**
        ```c
        printf(“this is a %s, with a number %d, and address %08x”,a,b,&c);
        ```
    :   * **Assumptions:** The Stack grows downwards towards lower addresses and arguments are pushed in reverse on the stack, also it operates on LIFO “last in first out” bases.
    :   ![img](/main_files/web_dev/images/s1.png){: width="47%"} 
    :   * What happens to the stack when a format string is specified with no corresponding variable on stack??!!
    :   ![img](/main_files/web_dev/images/s2.png){: width="57%"} 
    :   It will start to pop data off the stack from where the variables should have been located.

8. **Format Strings Protocol:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents38} 
    * The padding parameters to format specifiers are used to control the number of bytes output.
    * The ```%x``` token is used to pop bytes from the stack until the beginning of the format string itself is reached.
    * The start of the format string is crafted to contain the address that the %n format token can then overwrite with the address of the malicious code to execute.

***

## Integer Conversion Vulnerabilities
{: #content4}

1. **What?**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41}
    :   A vulnerability that occurs when an arithmetic operation attempts to create a numeric value that is outside of the range that can be represented with a given number of bits – either larger than the maximum or lower than the minimum representable value.

    :   > The most common result of an overflow is that the least significant representable bits of the result are stored; the result is said to wrap around the maximum.

2. **How?**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} 
    :   An attacker can use the representation mismatch to input a very large number, effectively, overdlowing the buffer.

3. **Why? (Who's at Risk?)**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43} 
    :   The C compiler won’t warn about the type mismatch between signed int and unsigned int; it silently inserts an implicit cast.

4. **Example:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents44} \\
    1. In this example, the prototype for ```memcpy()``` is: 
        ```c
        void *memcpy(void *dest, const void *src, size_t n); 
        ```
        And the definition of ```size_t``` is: 
        ```c
        typedef unsigned int size_t 
        ```
        If the attacker provides a negative value for len, the if statement won’t notice anything wrong, and ```memcpy()``` will be executed with a negative third argument.  
        C will cast this negative value to an unsigned int and it will become a very large positive.
        ```c
            char buf[80];
            void vulnerable() {
                int len = read_int_from_network();
                char *p = read_string_from_network();
                if (len > 80) {
                    error("length too large: bad dog, no cookie for you!");
                    return;
                }
                memcpy(buf, p, len);
            }
        ```
        > Thus ```memcpy()``` will copy a huge amount of memory into buf, overflowing the buffer.

    2. In this example, the code seems to avoid buffer overflow problems (by allocating 5 more bytes than necessary). But, there is a subtle problem: len+5 can wrap around if len is too large. For instance, if ```len = 0xFFFFFFFF```, then the value of ```len+5``` is 4 (on 32-bit platforms).  
    In this case, the code allocates a 4-byte buffer and then writes a lot more than 4 bytes into it: a classic buffer overflow.
        ```c
            void vulnerable() {
                size_t len;
                char *buf;
                len = read_int_from_network();
                buf = malloc(len+5);
                read(fd, buf, len);
                ...
            }
        ```

***

## Other Vulnerabilities
{: #content5}

1. **Dangling Pointers:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents51} 
    :   A pointer into a memory region that has been freed and is no longer valid.

2. **Double-Free Bugs:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents52} 
    :   Happens when a dynamically allocated object is explicitly freed multiple times

3. **Arc Injection:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents53} 
    :   By injecting malicious data that existing instructions later operates on, an attacker can still manipulate the execution path.
    :   > Recall that the ```ret``` instruction is equivalent to ```popl %eip```   
        > i.e., it writes the top of the stack into the program counter.

***

## Defending against Memory-Vulnerabilities
{: #content6}

1. **Secure Coding Practices:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents61}
    :   In general, before performing any potentially unsafe operation, we can write some code to check (at runtime) whether the operation is safe to perform and abort if not.

    * **Examples:** 
        1. This code ensures that the array access will be within bounds.  
        Instead of: 
        ```c
        char digit_to_char(int i) { // BAD
            char convert[] = "0123456789";
            return convert[i];
        }
        ```
        Write: 
        ```c
            char digit_to_char(int i) { // BETTER
                char convert[] = "0123456789";
                if (i < 0 || i > 9)
                    return "?"; // or, call exit()
                return convert[i];
            }
        ```
        2. The latter is better, because ```strlcpy(d,s,n)``` takes care to avoid writing more than ```n``` bytes into the buffer ```d```.  
        Instead of: 
        ```c
        char buf[512];
        strcpy(buf, src); // BAD
        ```
        Write: 
        ```c
        char buf[512];
        strlcpy(buf, src, sizeof buf); // BETTER
        ```
        > Basically, when calling library functions, we can use a library function that incorporates these kinds of checks.

        3. The latter is better, because ```strlcpy(d,s,n)``` takes care to avoid writing more than ```n``` bytes into the buffer ```d```.  
        Instead of: 
        ```c
        char buf[512];
        sprintf(buf, src); // BAD
        ```
        Write: 
        ```c
        char buf[512];
        snprintf(buf, sizeof buf, src); // BETTER
        ```
        > Basically, when calling library functions, we can use a library function that incorporates these kinds of checks.

        4. Instead of using ```gets()```, we can use ```fgets()```.

    * **What can we check?**
        1. Array indices are in-bounds before using them.
        2. Pointers are non-null and in-bounds before dereferencing them.
        3. Integer addition and multiplication won’t overflow or wrap around before performing the operation.
        4. Integer subtraction won’t underflow before performing it.
        5. Objects haven’t already been de-allocated before freeing them.
        6. Memory is initialized before being used.

2. **Defensive Programming:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents62}
    :   * It is a form of defensive design intended to ensure the continuing function of a piece of software under unforeseen circumstances.  
        > Defensive programming is like defensive driving:  
        > > The idea is to avoid depending on anyone else around you, so that if anyone else does something unexpected, you won’t crash.  

    :   * Defensive programming means that each module takes responsibility for checking the validity of all inputs sent to it.
        > Even if you “know” that your callers will never send you a NULL pointer, you check for NULL anyway (because it might change).

    :   * **Issues:** It shouldn't be used as the only means of defence because it assumes that the programmer will make no errors.


3. **Better Languages and Libraries:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents63}
    :   Languages and libraries can help avoid memory-safety vulnerabilities by eliminating the opportunity for programmer mistakes.
    :   * **For instance:**  
            1. Java performs automatic bounds-checking on every array access, so programmer error cannot lead to an array bounds violation.
            2. Java provides a String class with methods for many common string operations that are memory-safe.
            > The method itself performs all necessary runtime checks and resizes all buffers as needed to ensure there is enough space for strings.
            3. C++ provides a safe string class.

4. **Runtime Checks:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents64}
    :   Compilers and other tools can reduce the burden on programmers by automatically introducing runtime checks at every potentially unsafe operation, so that programmers do not have to do so explicitly.

    :   * **Issues with adding Runtime Checks to C:**  
            1. Automatic bounds-checking for C/C++ has a non-trivial performance overhead.
            2. Legacy code must be recompilied; thus, must be available and must be not too "out-dated" so that it DOES compile.

5. **Static Analysis:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents65}
    :   Static analysis is a technique for scanning source code to try to automatically detect potential bugs.  
    :   You can think of static analysis as runtime checks, performed at compile time:  
        the static analysis tool attempts to predict whether there exists any program execution under which a runtime check would fail, and if it finds any, it warns the programmer.

    :   * **Advantages:**  
            1. It can detect bugs proactively, at development time, so that they can be fixed before the code has been shipped.
            2. Makes fixing bugs cheaper because, the earlier a bug is found, the cheaper it can be to fix.

    :   * **Issues with adding Runtime Checks to C:**  
            1. **Fundamental Theorem:** detecting security bugs can be shown to be undecidable (like the Halting Problem).  
             So it follows that any static analysis tool will either miss some bugs (false negatives), or falsely warn about code that is correct (false positives), or both.
            
            2. **They make Errors:** from above.

    :   * **Tradeoff:** the effectiveness of a static analysis tool is determined by its false negative rate and false positive rate; these two can often be traded off against each other.
        >  At one extreme are verification tools, which are guaranteed to be free of false negatives: 
        > > if your code does not trigger any warnings, then it is guaranteed to be free of bugs (at least, of the sort of bugs that the tool attempts to detect).   
     
            > In practice, most developers accept a significant rate of false negatives in exchange for finding some relevant bugs, without too many false positives.

6. **Testing:**{: style="color: SteelBlue  "}{: .bodyContents6 #bodyContents66}
    :   * **Fuzz testing:** is one simple form of security testing. Fuzz testing involves testing the program with random inputs and seeing if the program exhibits any sign of failure. 
        > Usually, fuzz testing involves generating many inputs:   
        > e.g., hundreds of thousands or millions. 

            > Fuzz testing is popular in industry today because it is cheap, easy to apply, and somewhat effective at finding some kinds of bugs.   
    :   * **Aspects of Testing for Security:**  
            1. **Test Generation:** We need to find a way to generate test cases, so that we can run the program on those test cases.
                * **Random inputs:** Construct a random input file, and run the program on that input.  
                The file is constructed by choosing a totally random sequence of bytes, with no structure.
                * **Mutated inputs:** Start with a valid input file, randomly modify a few bits in the file, and run the program on the mutated input.
                * **Structure-driven input generation:** Taking into account the intended format of the input, devise a program to independently “fuzz” each field of the input file.  
                > For instance, if we know that one part of the input is a string, generate random strings (of random lengths, with random characters, some of them with % signs to try to trigger format string bugs, some with funny Unicode characters, etc.).   
                > If another part of the input is a length, try random integers, try a very small number, try a very large number, try a negative number (or an integer whose binary representation has its high bit set).  

                    > One issue with _random inputs_ is that, if the input has a structured format, then it is likely that a random input file will not have the proper format and thus will be quickly rejected by the program, leaving much of the code uncovered and untested.  
                    > > The other two  approaches address this problem.

            2. **Bug Detection:** We need a way to detect whether a particular test case revealed a bug in the program.
