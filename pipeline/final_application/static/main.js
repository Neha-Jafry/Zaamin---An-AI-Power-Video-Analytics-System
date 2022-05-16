// get references to the canvas and context
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let offsetX = canvas.offsetLeft;
let offsetY = canvas.offsetTop;
let scrollX = canvas.scrollLeft;
let scrollY = canvas.scrollTop;
// style the context
let isDown = false;
let startX = 0;
let startY = 0;
let mouseX = 0;
let mouseY = 0;
let submitButton = document.getElementById("button");

let rectangles= []
let imgw = 1;
let imgh = 1;
let width_new = 720;
let height_new = 720;
ctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
ctx.lineWidth = 3;
function LoadImg(src) {
  //Loading of the home test image - img1
  let img1 = new Image();

  //drawing of the test image - img1
  img1.onload = function () {
    //draw background image
	imgw = img1.width;
	imgh = img1.height;
	    canvas.width = 720;
      width_new = 720;
height_new = Math.floor(canvas.width * (img1.height/img1.width));
	    canvas.height = height_new;
	offsetX = canvas.offsetLeft;
	offsetY =canvas.offsetTop;
	scrollX = canvas.scrollLeft;
	scrollY = canvas.scrollTop;

    ctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
    ctx.lineWidth = 3;
  	ctx.save();
    ctx.drawImage(img1, 0, 0,width_new,height_new);

    //draw a box over the top
    // ctx.fillStyle = "rgba(0, 255, 0, 0.5)";
    // ctx.fillRect(0, 0, 500, 500);
  };
  img1.src = src;
  return img1;
}
let img = LoadImg("static/img_zone.png");

function handleMouseDown(e) {
  e.preventDefault();
  e.stopPropagation();
  if (!isDown) {
    // save the starting x/y of the rectangle
    startX = parseInt(e.clientX - offsetX);
    startY = parseInt(e.clientY - offsetY);
  }

  // set a flag indicating the drag has begun
  isDown = true;
}

function handleMouseUp(e) {
  e.preventDefault();
  e.stopPropagation();

  // the drag is over, clear the dragging flag
  let width = mouseX - startX;
  let height = mouseY - startY;

  if (isDown) {
    ctx.strokeRect(startX, startY, width, height);
    rectangles.push({x:startX,y:startY,w:width,h:height})
  }
  isDown = false;
}

function handleMouseOut(e) {
  e.preventDefault();
  e.stopPropagation();

  // the drag is over, clear the dragging flag
  isDown = false;
  ctx.save();
}

function handleMouseMove(e) {
  e.preventDefault();
  e.stopPropagation();

  // if we're not dragging, just return
  if (!isDown) {
    return;
  }

  // get the current mouse position
  mouseX = parseInt(e.clientX - offsetX);
  mouseY = parseInt(e.clientY - offsetY);
  if (isDown) {
	offsetX = canvas.offsetLeft;
	offsetY =canvas.offsetTop;
	scrollX = canvas.scrollLeft;
	scrollY = canvas.scrollTop;

    // iterate through all rectangles
    ctx.drawImage(img, 0, 0,width_new,height_new);
    for (let i = 0; i < rectangles.length; i++) {
      // if the mouse is inside the rectangle, draw a highlight
      console.log(rectangles[i])
        ctx.strokeRect(rectangles[i].x, rectangles[i].y, rectangles[i].w, rectangles[i].h);
      }
    let width = mouseX - startX;
    let height = mouseY - startY;    
      ctx.strokeRect(startX, startY, width, height);
  }
  // Put your mousemove stuff here

  // clear the canvas
  ctx.restore();

  // calculate the rectangle width/height based
  // on starting vs current mouse position
  // var width = mouseX - startX;
  // var height = mouseY - startY;

  // draw a new rect from the start position
  // to the current mouse position
  // ctx.strokeRect(startX, startY, width, height);
}

// listen for mouse events
canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("mouseup", handleMouseUp);
canvas.addEventListener("mouseout", handleMouseOut);
canvas.addEventListener("mousemove", handleMouseMove);

submitButton.addEventListener("click",function(){
  let xhr = new XMLHttpRequest();
  let newrectangles = []
	for (const rect of rectangles) { 
let newrectangle = {x:Math.floor(rect.x*(imgw/width_new)),y:Math.floor(rect.y*(imgh/height_new)),w:Math.floor(rect.w*(imgw/width_new)),h:Math.floor(rect.h*(imgh/height_new))}
newrectangles.push(newrectangle)
}
  xhr.open("POST", "/submitRectangles", true);
  console.log(newrectangles);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(JSON.stringify(newrectangles));
})
// canvas.mousedown(function (e) {
//     handleMouseDown(e);
// });
// canvas.mousemove(function (e) {
//     handleMouseMove(e);
// });
// canvas.mouseup(function (e) {
//     handleMouseUp(e);
// });
// canvas.mouseout(function (e) {
//     handleMouseOut(e);
// });
