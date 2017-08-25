---
layout: page
title: TODO [ACADEMIC]
permalink: /projects/TODO1
---

<a href="#" class="clear" onclick="clearer(event)">Clear</a>

<h3> TODO list:</h3>
<ol id="par">
</ol>

<form id="frm" autocomplete="off">
  <label for="todo"></label>
  <input type="text" id="todo" name="todo">
</form>


<script type="text/javascript">
var i = parseInt(localStorage["index"]) + 1 || 1;

$(document).ready(function() {

    for (var i = 1; i < 100; i++) {
        // var element = paragraphs[i]; // 
        var $todo = localStorage[String(i)] || 0;
        console.log(i + ": " +  $todo);
        if ($todo === 0 || $todo === "0") {
            continue;
        } else {
        var $ul = $("#par");
        var $li = $("<li>");
        $li.attr("id", String(i));
        $li.text($todo);
        $li.click(remover);
        $ul.append($li);
        }
    }
});

function clearer(event) {
    event.preventDefault();
    localStorage.clear();
    location.reload();
 };


function remover(event) {
    localStorage.removeItem(String($(this).attr("id")));
    localStorage["index"] = String(parseInt(localStorage["index"])-1);
    $(this).remove();
 };




 $("#frm").on("submit", function(event) {
    event.preventDefault();
    console.log("i: ", i)
    var $todo = $(this).find('[name=todo]').val();
    // Store the value of the input with name='age'
    var $ul = $("#par");

    var $li = $("<li>");
    $li.attr("id", String(i));
    $li.text($todo);
    $li.click(remover);
    $ul.append($li);
    localStorage[String(i)] = $todo;
    localStorage["index"] = i;
    i++;
    $("#todo").val("");
 });


</script>




<style> 

ol {
    padding: 5;
    border: 1.5px solid #ccc;
    border-color: darkblue;
    border-radius: 3px;
}

input:focus {
  outline: none;
}

.clear {
    -moz-box-shadow:inset 0px 0px 15px 3px #0c2142;
    -webkit-box-shadow:inset 0px 0px 15px 3px #0c2142;
    box-shadow:inset 0px 0px 15px 3px #0c2142;
    background:-webkit-gradient(linear, left top, left bottom, color-stop(0.05, #3679e3), color-stop(1, #417987));
    background:-moz-linear-gradient(top, #3679e3 5%, #417987 100%);
    background:-webkit-linear-gradient(top, #3679e3 5%, #417987 100%);
    background:-o-linear-gradient(top, #3679e3 5%, #417987 100%);
    background:-ms-linear-gradient(top, #3679e3 5%, #417987 100%);
    background:linear-gradient(to bottom, #3679e3 5%, #417987 100%);
    filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#3679e3', endColorstr='#417987',GradientType=0);
    background-color:#3679e3;
    -moz-border-radius:25px;
    -webkit-border-radius:25px;
    border-radius:25px;
    border:1px solid #1f2f47;
    display:inline-block;
    cursor:pointer;
    color:#ffffff;
    font-family:Impact;
    font-size:15px;
    font-weight:bold;
    padding:6px 13px;
    text-decoration:none;
    text-shadow:0px 1px 0px #263666;
}
.clear:hover {
    background:-webkit-gradient(linear, left top, left bottom, color-stop(0.05, #417987), color-stop(1, #3679e3));
    background:-moz-linear-gradient(top, #417987 5%, #3679e3 100%);
    background:-webkit-linear-gradient(top, #417987 5%, #3679e3 100%);
    background:-o-linear-gradient(top, #417987 5%, #3679e3 100%);
    background:-ms-linear-gradient(top, #417987 5%, #3679e3 100%);
    background:linear-gradient(to bottom, #417987 5%, #3679e3 100%);
    filter:progid:DXImageTransform.Microsoft.gradient(startColorstr='#417987', endColorstr='#3679e3',GradientType=0);
    background-color:#417987;
}
.clear:active {
    position:relative;
    top:1px;
}

input[type=text] {
    width: 100%;
    padding: 12px 20px;
    margin: 8px 0;
    box-sizing: border-box;
    border: 2px solid grey;
    /*border-color: #8b00ff;*/
    border-radius: 4px;
}

</style>