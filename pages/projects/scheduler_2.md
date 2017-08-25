---
layout: page
title: TODO [MISC]
permalink: /projects/TODO2
---

<button id="clear" onclick="clearer()">clear</button>

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

function clearer() {
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

input[type=text] {
    width: 100%;
    padding: 12px 20px;
    margin: 8px 0;
    box-sizing: border-box;
    border: 2px solid grey;
    /*border-color: #8b00ff;*/
    border-radius: 4px;
}

input:focus {
  outline: none;
}

</style>