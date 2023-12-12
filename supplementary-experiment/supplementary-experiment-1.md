<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-cly1{text-align:left;vertical-align:middle}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-cly1" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="2">RCV1-v2</th>
    <th class="tg-nrix" colspan="2">Reuters-21578</th>
    <th class="tg-nrix" colspan="2">AAPD</th>
    <th class="tg-nrix" colspan="2">freecode</th>
    <th class="tg-nrix" colspan="2">EUR-Lex</th>
  </tr>
  <tr>
    <th class="tg-nrix">P@1</th>
    <th class="tg-nrix">Micro-F1@5</th>
    <th class="tg-wa1i"><span style="font-weight:400;font-style:normal">P@1</span></th>
    <th class="tg-wa1i"><span style="font-weight:400;font-style:normal">Micro-F1@5</span></th>
    <th class="tg-wa1i"><span style="font-weight:400;font-style:normal">P@1</span></th>
    <th class="tg-wa1i"><span style="font-weight:400;font-style:normal">Micro-F1@5</span></th>
    <th class="tg-wa1i"><span style="font-weight:400;font-style:normal">P@1</span></th>
    <th class="tg-nrix"><span style="font-weight:400;font-style:normal">Micro-F1@5</span></th>
    <th class="tg-nrix"><span style="font-weight:400;font-style:normal">P@1</span></th>
    <th class="tg-nrix"><span style="font-weight:400;font-style:normal">Micro-F1@5</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-cly1">Bert</td>
    <td class="tg-nrix">74.442</td>
    <td class="tg-nrix"><span style="font-weight:400;font-style:normal">37.269</span></td>
    <td class="tg-nrix">74.530</td>
    <td class="tg-nrix">31.606</td>
    <td class="tg-nrix">75.000</td>
    <td class="tg-nrix">47.625</td>
    <td class="tg-nrix">40.890</td>
    <td class="tg-nrix">28.780</td>
    <td class="tg-nrix">58.201</td>
    <td class="tg-nrix">26.519</td>
  </tr>
  <tr>
    <td class="tg-cly1">Bert + FL</td>
    <td class="tg-nrix">78.850</td>
    <td class="tg-nrix">39.357</td>
    <td class="tg-nrix">74.261</td>
    <td class="tg-nrix">31.219</td>
    <td class="tg-nrix">75.565</td>
    <td class="tg-nrix">45.976</td>
    <td class="tg-nrix">40.834</td>
    <td class="tg-nrix">28.276</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
  </tr>
  <tr>
    <td class="tg-cly1">Bert + FL<br>+ Semantic Label</td>
    <td class="tg-nrix">25.867</td>
    <td class="tg-nrix">7.701</td>
    <td class="tg-nrix">37.03</td>
    <td class="tg-nrix">12.384</td>
    <td class="tg-nrix">11.864</td>
    <td class="tg-nrix">8.755</td>
    <td class="tg-nrix">8.326</td>
    <td class="tg-nrix">2.688</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
  </tr>
  <tr>
    <td class="tg-cly1">Bert + ours</td>
    <td class="tg-wa1i">79.026</td>
    <td class="tg-wa1i">40.262</td>
    <td class="tg-wa1i">77.352</td>
    <td class="tg-wa1i">33.477</td>
    <td class="tg-wa1i">76.695</td>
    <td class="tg-wa1i">48.214</td>
    <td class="tg-wa1i">45.579</td>
    <td class="tg-wa1i">31.016</td>
    <td class="tg-wa1i">71.107</td>
    <td class="tg-wa1i">35.699</td>
  </tr>
</tbody>
</table>