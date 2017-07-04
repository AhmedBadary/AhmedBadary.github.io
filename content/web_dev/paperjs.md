---
layout: NotesPage
title: Paper.JS
permalink: /work_files/web_dev/PaperJS
prevLink: /work_files/web_dev.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Basic Interaction](#content1)
  {: .TOC1}
  * [Paths](#content2)
  {: .TOC2}
  * [Animations](#content3)
  {: .TOC3}
  * [Events](#content4)
  {: .TOC4}
</div>

***
***

## Basic Interaction
{: #content1}

1. **Create Vectors:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11} \\
    ```javascript
    var vector = new Point(X, Y);
    ```
    > Note\\
    > This is *PaperScript*

2. **Create Paths:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} \\
    ```javascript
    var path = new Path();
    ```

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents14} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents15} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents16} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents17} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents18} \\

***

## Paths
{: #content2}

1. **Creating Paths:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21} \\
    ```javascript
    var path = new Path();
    ```

2. **Coloring the Path:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22} \\
    ```javascript
    path.strokeColor = ‘red’;
    ```

3. **Shaped Path Objects:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents23} \\
    <button>Click for list of Shapes</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * **Line:** `Path.Line(from, to)`
    * **Line:** `Path.Line(object)`
    * **Circle:** `Path.Circle(center, radius)`
    * **Circle:** `Path.Circle(object)`
    * **Rectangle:** `Path.Rectangle(rectangle[, radius])`
    * **Rectangle:** `Path.Rectangle(point, size)`
    * **Rectangle:** `Path.Rectangle(from, to)`
    * **Rectangle:** `Path.Rectangle(object)`
    * **Ellipse:** `Path.Ellipse(rectangle)`
    * **Ellipse:** `Path.Ellipse(object)`
    * **Arc:** `Path.Arc(from, through, to)`
    * **Arc:** `Path.Arc(object)`
    * **RegularPolygon:** `Path.RegularPolygon(center, sides, radius)`
    * **RegularPolygon:** `Path.RegularPolygon(object)`
    * **Star:** `Path.Star(center, points, radius1, radius2)`
    * **Star:** `Path.Star(object)`
    {: hidden=''}

4. **Methods:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents24} \\
    <button>Click for list of Methods</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * [**Add:**](http://paperjs.org/reference/path/#add-segment){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#add-segment">`add(segment)`</a>
        <div markdown="1"> </div>
    * [**Insert:**](http://paperjs.org/reference/path/#insert-index-segment){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#insert-index-segment">`insert(index, segment)`</a>
        <div markdown="1"> </div>
    * [**addSegments:**](http://paperjs.org/reference/path/#addsegments-segments){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#addsegments-segments">`addSegments(segments)`</a>
        <div markdown="1"> </div>
    * [**insertSegments:**](http://paperjs.org/reference/path/#insertsegments-index-segments){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#insertsegments-segments">`insertSegments(index, segments)`</a>
        <div markdown="1"> </div>
    * [**removeSegment:**](http://paperjs.org/reference/path/#removesegment-index){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#removesegment-index">`removeSegment(index)`</a>
        <div markdown="1"> </div>
    * [**removeSegments:**](http://paperjs.org/reference/path/#removesegment){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#removesegments">`removeSegments()`</a>
        <div markdown="1"> </div>
    * [**removeSegments:**](http://paperjs.org/reference/path/#removesegment-from){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#removesegments-from">`removeSegments(from[, to])`</a>
        <div markdown="1"> </div>
    * [**hasHandles:**](http://paperjs.org/reference/path/#hashandles){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#hashandles">`hasHandles()`</a>
        <div markdown="1"> </div>
    * [**clearHandles:**](http://paperjs.org/reference/path/#clearhandles){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#clearhandles">`clearHandles()`</a>
        <div markdown="1"> </div>
    * [**divideAt:**](http://paperjs.org/reference/path/#divideat-location){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#divideat-location">`divideAt(location)`</a>
        <div markdown="1"> </div>
    * [**splitAt:**](http://paperjs.org/reference/path/#splitat-location){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#splitat-location">`splitAt(location)`</a>
        <div markdown="1"> </div>
    * [**join:**](http://paperjs.org/reference/path/#join-path){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#join-path">`join(path[, tolerance])`</a>
        <div markdown="1"> </div>
    * [**reduce:**](http://paperjs.org/reference/path/#reduce-options){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#reduce-options">`reduce(options)`</a>
        <div markdown="1"> </div>
    * [**toShape:**](http://paperjs.org/reference/path/#toshape){: value="show" onclick="iframePopA(event)"}
        <a href="http://paperjs.org/reference/path/#toshape">`toShape([insert])`</a>
        <div markdown="1"> </div>
    {: hidden=""}

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents25} \\
    ```javascript

    ```

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents26} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents27} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents28} \\

***

## Animation
{: #content3}

1. **Function:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31} \\
   ```javascript 
    function onFrame(event) {
        // 60x/s
    }
    ```
2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} \\

5. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents35} \\

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents36} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents37} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents38} \\

***

## Events
{: #content4}

1. **MouseDown:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41} \\
   ```javascript 
    Function onMouseDown(event) {
    }
    ```
2. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42} \\

3. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43} \\

4. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents44} \\

5. **Event Properties:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents45} \\
    <button></button>{: .showText value="show"
     onclick="showTextPopHide(event);"}
    * **Event Position:** `event.point;`

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents46} \\

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents47} \\

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents48} \\
