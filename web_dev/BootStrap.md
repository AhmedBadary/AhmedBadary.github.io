---
layout: NotesPage
title: BootStrap
permalink: /work_files/web_dev/BootStrap
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Including the Library and the General Structure](#content1)
  {: .TOC1}
  * [The Grid System](#content2)
  {: .TOC2}
  * [Form Processing](#content3)
  {: .TOC3}
</div>

***
***

## Including the Library and the General Structure
{: #content1}
1. **Including BootStrap [CDN]:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
```html
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp" crossorigin="anonymous">
```
2. **General Structure:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} \\
    So, to create the layout you want, create a container `<div class="container">`. Next, create a row `<div class="row">`. Then, add the desired number of columns (tags with appropriate `.col-*-*` classes). Note that numbers in `.col-*-*` should always add up to 12 for each row.

3. **The Container:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}
    * **Fixed Width:** `class="container"`
    * **Full Width:** `class="container-fluid"`


***

## The Grid System
{: #content2}
1. **A BootStrap Grid Structure:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21}
```html
    <div class="container">
      <div class="row">
        <div class="col-*-*"></div>
      </div>
      <div class="row">
        <div class="col-*-*"></div>
        <div class="col-*-*"></div>
        <div class="col-*-*"></div>
      </div>
      <div class="row">
        ...
      </div>
    </div>
```
2. **Rows:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22}
    Use rows to create horizontal groups of columns.
3. **Grid Options:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23}\\
    `.col-xs-` ,   `.col-sm-`  ,  `.col-md-`  ,  `.col-lg-`
4. **Columns:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents24}
    * **Number of Columns:** 12
    * **Column Offset:** `class="col-lg-offset-#num"`
5. **Clear Floats:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents25}\\
    To prevent strange wrapping with uneven content \\
    `<div class="clearfix visible-xs"></div>`
6. **A BootStrap Grid Structure:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents26}
```html
```
7. **Alignments:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}
    * **Left Align:** `class="pull-left"`
    * **Right Align:** `class="pull-right"`