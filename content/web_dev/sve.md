---
layout: NotesPage
title: Security Vulnerabilities <br /> Examples
permalink: /work_files/web_dev/security/sve
prevLink: /work_files/web_dev/security.html
---


<div markdown="1" class = "TOC">
# Table of Contents

  * [Buffer Overflow](#content1)
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

## Buffer Overflow
{: #content1}

1. **Disc.2:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11} \\
    ```c
        struct food {
            char name[1 0 2 4];
            int c a l o r i e s;
        };
        /* Evaluate a shopping basket with at most 32 food items.
        Returns the number of low-calorie items, or -1 on a problem. */
            int evalbasket(struct food basket[], size_t n) {
                struct food good[32];
                char bad[1024], cmd[1024];
                int i, total= 0, ngood = 0, size_bad = 0;

                if (n > 32) return -1;

                for(i = 0; i <= n; ++i) {
                    if (basket[i].calories<100)
                        good[ngood++] = basket[i];
                    else if(basket[i].calories>500) {
                        size_t len = strlen(basket[i].name);
                        snprintf(bad + size_bad, len, "%s", basket[i].name);
                        size_bad += len;
                    }

                    total+=basket[i].calories;
                }

                if (total > 2500) {
                    const char* fmt = "health-factor--calories %d --bad-items %s";
                    fprintf(stderr, "lotsofcalories!");
                    snprintf(cmd, sizeof cmd, fmt, total, bad);
                    system(cmd);
                }

                return ngood;
            }
    ```
    * **Where:**
        * ```strlen``` calculates the length of a string, not including the terminating ‘\0’ character.
        * ```snprintf(buf, len, fmt, . . . )``` works like printf, but instead writes to buf, and won’t write more than len - 1 characters. It terminates the characters written with a ```‘\0’```.
        * system runs the shell command given by its first argument.

    * **Solution:**
        1. Line 15 has a fencepost error: the conditional test should be i < n rather than i <= n. The test at line 13 assures that n doesn’t exceed 32, but if it’s equal to 32, and if all of the items in basket are “good”, then the assignment at line 17 will write past the end of good, representing a buffer overflow vulnerability.
        2. At line 20, there’s an error in that the length passed to snprintf is supposed to be available space in the buffer (which would be sizeof bad - size bad), but instead it’s the length of the string being copied (along with a blank) into the buffer. Therefore by supplying large names for items in basket, the attacker can write past the end of bad at this point, again representing a buffer overflow vulnerability.
        3. At line 31, a shell command is run based on the contents of cmd, which in turn includes values from bad, which in turn is derived from input provided by the attacker. That input could include shell command characters such as pipes (‘\|’) or command separators (‘;’), facilitating _command injection_.

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents14} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents15} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents16} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents17} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents18} \\

***

## SECOND
{: #content2}

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21} \\

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents24} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents25} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents26} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents27} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents28} \\

***

## THIRD
{: #content3}

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31} \\

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents35} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents36} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents37} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents38} \\

***

## FOURTH
{: #content4}

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41} \\

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents44} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents45} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents46} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents47} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents48} \\

***

## FIFTH
{: #content5}

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents51} \\

2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents52} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents53} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents54} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents55} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents56} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents57} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents58} \