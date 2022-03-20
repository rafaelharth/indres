"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Common HTML visualization utilities
"""

import numpy as np
from PIL import Image

import settings

FILTERBOX = """
<input type="text" placeholder="Filter by unit" id="filterField">
<button type="button" class="filterby">Filter</button>
"""
HTML_SORTHEADER = """
<div class="sortheader">
sort by
<span class="sortby currentsort" segmentation_data-index="0">label</span>
{}
</div>
"""


def wrap_image(html, wrapper_classes=None, infos=None):
    """
    Wrap an image tag with image wrapper stuff
    """
    if wrapper_classes is None:
        wrapper_classes = []
    if infos is None:
        infos = []

    wrc = " " + " ".join(wrapper_classes)
    info_htmls = [f"<p>{i}</p>" for i in infos]

    wr_htmls = [
        f'<div class="img-wrapper{wrc}">',
        f'<div class="img-background">',
        html,
        f"</div>",
        f'<div class="img-wrapper-info">',
        *info_htmls,
        f"</div>",
        f"</div>",
    ]
    return "".join(wr_htmls)


def create_mask(index, neuron_i, inspected_neurons, thresholds, imsize=settings.IMG_SIZE):
    relevant_neurons = inspected_neurons[index, neuron_i]
    threshold = thresholds[neuron_i]
    mask = (relevant_neurons > threshold).astype(np.uint8) * 255
    mask = np.clip(mask, 50, 255)
    mask = Image.fromarray(mask).resize((imsize, imsize), resample=Image.NEAREST)
    # All black
    mask_alpha = Image.fromarray(np.zeros((imsize, imsize), dtype=np.uint8), mode="L")
    mask_alpha.putalpha(mask)
    return mask_alpha


def to_labels(
    unit, contr, weight, prev_unit_names, uname=None, label_class="contr-label"
):
    """
    :param contr: binary ndarray of curr units x
    prev units; 1 if prev unit contributets to
    curr unit
    :param weight: continuous ndarray of curr units x prev units; contains the actual weights from which contr was binarized
    :param prev_unit_names: map from units to their names, *one-indexed*
    :param uname: if provided, use this as the unit name
    """

    def get_tally_label(u):
        if u not in prev_unit_names:
            return "unk"
        return prev_unit_names[u]["label"]

    contr = np.where(contr[unit])[0]
    weight = weight[unit, contr]
    if uname is None:
        uname = unit
    contr_labels = [
        f'<span class="label {label_class}" segmentation_data-unit="{u}" segmentation_data-uname="{uname}">{u} ({get_tally_label(u)}, {w:.3f})</span>'
        for u, w in sorted(zip(contr, weight), key=lambda x: x[1], reverse=True)
    ]
    contr_label_str = ", ".join(contr_labels)
    contr_url_str = ",".join(map(str, [c for c in contr]))
    return contr_url_str, contr_label_str, contr


def get_sortheader(names):
    return HTML_SORTHEADER.format(
        "\n".join(
            f'<span class="sortby" segmentation_data-index="{i}">{name}</span>'
            for i, name in enumerate(names, start=1)
        )
    )


HTML_PREFIX = """
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.mask-img {
    -webkit-mask-size: contain;
}
.label:hover {
    background-color: yellow;
    color: black;
    font-weight: bold;
}
button {
    cursor: pointer;
}
.bluespan {
    color: blue;
}
.redspan {
    color: red;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .l_name {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
    width: 1100px;
    display: inline-block;
    white-space: nowrap;
    overflow-x: scroll;
}
.img-wrapper {
}
.correct {
background-color: #90EE90;
}
.incorrect {
background-color: #FF7F7F;
}
.contr {
}
.contributors a {
    color: green;
}
.inhibitors a {
    color: red;
}
.midrule {
    margin-top: 1em;
    margin-bottom: 0.25em;
}
.unit {
  width: 1200px;
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  border: 3px solid white;
  text-align: center;
  display: inline-block;
}
.img-background {
    background-color: #333;
    display: inline-block;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.prediction-correct {
    color: green;
}
.prediction-incorrect {
    color: red;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  // text-decoration: none;
  font-weight: bold;
  // cursor: default;
}
.sort-up::after {
  content: " - (up)";
}
.sort-down::after {
  content: " - (down)";
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
"""

HTML_SUFFIX = """
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             segmentation_data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <div class="img-background">
            <img class="fullsize img-fluid" src="//:0">
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$('img:not([segmentation_data-nothumb])[src]').wrap(function() {
  var result = $('<a segmentation_data-toggle="lightbox">')
  result.attr('href', $(this).attr('src'));
  result.attr('segmentation_data-style', $(this).attr('style'));
  var caption = $(this).closest('figure').find('figcaption').text();
  if (!caption && $(this).closest('.citation').length) {
    caption = $(this).closest('.citation').text();
  }
  if (caption) {
    result.attr('segmentation_data-footer', caption);
  }
  var title = $(this).attr('title');
  if (!title) {
    title = $(this).closest('td').find('.unit,.score').map(function() {
      return $(this).text(); }).toArray().join('; ');
  }
  if (title) {
    result.attr('segmentation_data-title', title);
  }
  return result;
});
$(document).on('click', '[segmentation_data-toggle=lightbox]', function(event) {
    $('#lightbox img').attr('src', $(this).attr('href'));
    var maskStyle = $(this).segmentation_data('style');
    if (maskStyle != undefined) {
        $('#lightbox img').attr('style', $(this).segmentation_data('style'));
    } else {
        $('#lightbox img').removeAttr('style');
    }
    $('#lightbox .modal-title').text($(this).segmentation_data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).segmentation_data('footer') ||
       $(this).closest('.unit').find('.info').text());
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('keydown', function(event) {
    $('#lightbox').modal('hide');
});
$(document).on('click', '.sortby', function(event) {
    if ($(this).hasClass('currentsort')) {
        if ($(this).hasClass('sort-up')) {
            // switch to negative sort
            var dir = -1;
            $(this).removeClass('sort-up');
            $(this).addClass('sort-down');
        } else {
            // switch to positive sort
            var dir = 1;
            $(this).removeClass('sort-down');
            $(this).addClass('sort-up');
        }
    } else {
        // default to positive sort
        var dir = 1;
        $('.sortby').removeClass('currentsort');
        $('.sortby').removeClass('sort-up');
        $('.sortby').removeClass('sort-down');
        $(this).addClass('currentsort');
        $(this).addClass('sort-up');
    }
    var sortindex = +$(this).segmentation_data('index');
    sortBy(sortindex, dir);
});
$(document).on('click', '.filterby', function(event) {
    var u = $('#filterField').val().split(',');
    var u = u.map(function(i) { return parseInt(i); });
    filterBy(u);
})
function sortBy(index, dir) {
  $('.unitgrid').find('.unit').sort(function (a, b) {
     return dir * (+$(a).eq(0).segmentation_data('order').split(' ')[index] -
            +$(b).eq(0).segmentation_data('order').split(' ')[index]);
  }).appendTo('.unitgrid');
}
function filterBy(units) {
  console.log(units);
  $('.unitgrid').find('.unit').filter(function (index, element) {
    // 0th is units
    if (units.length === 0 || units.includes(NaN)) {
        $(element).show();
        }
    else {
        if (units.includes(parseInt($(element).segmentation_data('order').split(' ')[2]))) {
            $(element).show();
        } else {
            $(element).hide();
        }
    }
  });
}

$(document).ready(function() {
    // Filter units
    var url = new URL(window.location.href);
    var u = url.searchParams.get('u');
    if (u != null) {
        var us = u.split(',');
        var us = us.map(function(i) { return parseInt(i); });
        filterBy(us);
    }
    // Highlight RFs when hovering over image
    $(document).on('mouseenter', '.contr-label',
        function(e) {
            var uname = $(this).segmentation_data('uname');
            var unit = $(this).segmentation_data('unit');
            $('.mask-img[segmentation_data-uname="' + uname + '"]').each(function(i, e) {
                var imfn = $(this).segmentation_data('imfn');
                var imalpha = imfn.replace('.jpg', '.png');
                var imalpha = 'images/mask-' + unit + '-' + imalpha;
                console.log('Loading ' + imalpha);
                $(this).css('-webkit-mask-image', 'url(' + imalpha + ')');
            });
        }
    );
    $(document).on('mouseleave', '.contr-label',
        function(e) {
            var uname = $(this).segmentation_data('uname');
            var unit = $(this).segmentation_data('unit');
            $('.mask-img[segmentation_data-uname="' + uname + '"]').each(function(i, e) {
                if ($(this).segmentation_data('masked')) {
                    var imfn = $(this).segmentation_data('imfn');
                    var imalpha = 'images/this-mask-' + uname + '-' + imfn;
                    console.log('Restoring ' + imalpha);
                    $(this).css('-webkit-mask-image', 'url(' + imalpha + ')');
                } else {
                    console.log('Clearing mask');
                    $(this).css('-webkit-mask-image', '');
                }
            });
        }
    );
});
</script>
</body>
</html>
"""