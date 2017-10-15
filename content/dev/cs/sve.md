---
layout: NotesPage
title: Security Vulnerabilities <br /> Examples
permalink: /work_files/dev/cs/sve
prevLink: /work_files/dev/cs.html
---


<div markdown="1" class = "TOC">
# Table of Contents

  * [Buffer Overflow](#content1)
  {: .TOC1}
  * [FOURTH](#content4)
  {: .TOC4}
</div>

***
***

## Buffer Overflow
{: #content1}

1. **Discussion:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11} \\
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


***

## Proving Memory Safety
{: #content4}

1. **Ex.1:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41} \\
    <button>Show Example</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    ![img](/main_files/cs/sve/41.png){: width="50%" hidden=""}

2. **Ex.2:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} \\
    <button>Show Example</button>{: .showText value="show"
    onclick="showTextPopHide(event);"}
    ![img](/main_files/cs/sve/42.png){: width="60%" hidden=""}