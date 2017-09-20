---
layout: NotesPage
title: HW 1
permalink: /work_files/dev/cs/hw
prevLink: /work_files/dev/cs.html
---

## Q.1)


There are no "slip days".

***

## Q.2)

No.

***

## Q.3)

1. Line 4 has a memory vulnerability. It is a buffer overflow vulnerability that could lead to a "Stack Smashing Attack".

2. The attacker can easily overflow the buffer and overwrite the return address that is saved after the function call to ```scanf```. He, then, can write over the return address with some other address that might contain some malicious code. 
Address: 0xBFFFFB4C

3. Add stack canaries between the local variables and the control. The canary will allow us to detect if an overflow happened and then we can just abort everything (Halt).

4. Yes. Because we can now detect if anything got overwritten, and thus we can easily dump the core (Halt the program) to stop any potential danger.

## Q.4)

Nothing.