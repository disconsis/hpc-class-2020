# Setup


### Compiler version (`g++ -v`)

    Using built-in specs.
    COLLECT_GCC=/usr/bin/g++
    COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-pc-linux-gnu/9.2.1/lto-wrapper
    Target: x86_64-pc-linux-gnu
    Configured with: /build/gcc/src/gcc/configure --prefix=/usr --libdir=/usr/lib ...
    Thread model: posix
    gcc version 9.2.1 20200130 (Arch Linux 9.2.1+20200130-2)


### Processor

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Model</td>
<td class="org-left">Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz</td>
</tr>


<tr>
<td class="org-left">Max Flop rate</td>
<td class="org-left">25.96 GFLOPs</td>
</tr>


<tr>
<td class="org-left">Max bandwidth</td>
<td class="org-left">35.76 GiB/s</td>
</tr>
</tbody>
</table>


# Q2. Matrix-matrix multiplication


## Questions

The optimal value of `BLOCK_SIZE` is 16.
The code achieves a maximum of 80% of peak flop rate for the smaller sized matrices.


## Timings

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">Dimension</th>
<th scope="col" class="org-right">Time - Blocked version</th>
<th scope="col" class="org-right">Time - OpenMP version</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">1024</td>
<td class="org-right">0.392636</td>
<td class="org-right">0.254664</td>
</tr>


<tr>
<td class="org-right">1072</td>
<td class="org-right">0.241596</td>
<td class="org-right">0.238849</td>
</tr>


<tr>
<td class="org-right">1120</td>
<td class="org-right">0.297218</td>
<td class="org-right">0.253248</td>
</tr>


<tr>
<td class="org-right">1168</td>
<td class="org-right">0.327683</td>
<td class="org-right">0.286684</td>
</tr>


<tr>
<td class="org-right">1216</td>
<td class="org-right">0.437334</td>
<td class="org-right">0.280523</td>
</tr>


<tr>
<td class="org-right">1264</td>
<td class="org-right">0.407966</td>
<td class="org-right">0.400364</td>
</tr>


<tr>
<td class="org-right">1312</td>
<td class="org-right">0.509102</td>
<td class="org-right">0.836481</td>
</tr>


<tr>
<td class="org-right">1360</td>
<td class="org-right">0.507661</td>
<td class="org-right">0.437908</td>
</tr>


<tr>
<td class="org-right">1408</td>
<td class="org-right">0.892849</td>
<td class="org-right">0.442123</td>
</tr>


<tr>
<td class="org-right">1456</td>
<td class="org-right">0.636359</td>
<td class="org-right">0.475716</td>
</tr>


<tr>
<td class="org-right">1504</td>
<td class="org-right">1.014875</td>
<td class="org-right">0.670422</td>
</tr>


<tr>
<td class="org-right">1552</td>
<td class="org-right">0.793876</td>
<td class="org-right">0.619685</td>
</tr>


<tr>
<td class="org-right">1600</td>
<td class="org-right">1.336579</td>
<td class="org-right">0.713375</td>
</tr>


<tr>
<td class="org-right">1648</td>
<td class="org-right">0.930474</td>
<td class="org-right">0.898537</td>
</tr>


<tr>
<td class="org-right">1696</td>
<td class="org-right">1.551997</td>
<td class="org-right">0.768537</td>
</tr>


<tr>
<td class="org-right">1744</td>
<td class="org-right">1.202396</td>
<td class="org-right">0.813551</td>
</tr>


<tr>
<td class="org-right">1792</td>
<td class="org-right">2.446570</td>
<td class="org-right">1.033018</td>
</tr>


<tr>
<td class="org-right">1840</td>
<td class="org-right">1.312623</td>
<td class="org-right">0.961884</td>
</tr>


<tr>
<td class="org-right">1888</td>
<td class="org-right">1.960448</td>
<td class="org-right">1.084041</td>
</tr>


<tr>
<td class="org-right">1936</td>
<td class="org-right">1.717503</td>
<td class="org-right">1.188402</td>
</tr>


<tr>
<td class="org-right">1984</td>
<td class="org-right">2.431242</td>
<td class="org-right">1.313678</td>
</tr>
</tbody>
</table>


# Q4. Jacobi/Gauss-Seidel smoothing


## Timings

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">Dimension</th>
<th scope="col" class="org-right">Threads</th>
<th scope="col" class="org-right">Time - Jacobi</th>
<th scope="col" class="org-right">Time - Gauss-Seidel</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">1000</td>
<td class="org-right">1</td>
<td class="org-right">0.016777</td>
<td class="org-right">0.0420625</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">2</td>
<td class="org-right">0.01272</td>
<td class="org-right">0.0300278</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">4</td>
<td class="org-right">0.0113041</td>
<td class="org-right">0.0254845</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">8</td>
<td class="org-right">0.0912515</td>
<td class="org-right">0.131496</td>
</tr>


<tr>
<td class="org-right">1000</td>
<td class="org-right">16</td>
<td class="org-right">0.0162705</td>
<td class="org-right">0.0315926</td>
</tr>


<tr>
<td class="org-right">2000</td>
<td class="org-right">1</td>
<td class="org-right">0.0759777</td>
<td class="org-right">0.147523</td>
</tr>


<tr>
<td class="org-right">2000</td>
<td class="org-right">2</td>
<td class="org-right">0.061264</td>
<td class="org-right">0.117789</td>
</tr>


<tr>
<td class="org-right">2000</td>
<td class="org-right">4</td>
<td class="org-right">0.0584821</td>
<td class="org-right">0.126328</td>
</tr>


<tr>
<td class="org-right">2000</td>
<td class="org-right">8</td>
<td class="org-right">0.130799</td>
<td class="org-right">0.154627</td>
</tr>


<tr>
<td class="org-right">2000</td>
<td class="org-right">16</td>
<td class="org-right">0.0544273</td>
<td class="org-right">0.12954</td>
</tr>


<tr>
<td class="org-right">5000</td>
<td class="org-right">1</td>
<td class="org-right">0.450877</td>
<td class="org-right">0.982183</td>
</tr>


<tr>
<td class="org-right">5000</td>
<td class="org-right">2</td>
<td class="org-right">0.356313</td>
<td class="org-right">0.789708</td>
</tr>


<tr>
<td class="org-right">5000</td>
<td class="org-right">4</td>
<td class="org-right">0.309172</td>
<td class="org-right">0.986109</td>
</tr>


<tr>
<td class="org-right">5000</td>
<td class="org-right">8</td>
<td class="org-right">0.321736</td>
<td class="org-right">1.12963</td>
</tr>


<tr>
<td class="org-right">5000</td>
<td class="org-right">16</td>
<td class="org-right">0.304599</td>
<td class="org-right">0.758378</td>
</tr>


<tr>
<td class="org-right">10000</td>
<td class="org-right">1</td>
<td class="org-right">1.68608</td>
<td class="org-right">3.50216</td>
</tr>


<tr>
<td class="org-right">10000</td>
<td class="org-right">2</td>
<td class="org-right">1.34041</td>
<td class="org-right">3.01577</td>
</tr>


<tr>
<td class="org-right">10000</td>
<td class="org-right">4</td>
<td class="org-right">1.21576</td>
<td class="org-right">3.10543</td>
</tr>


<tr>
<td class="org-right">10000</td>
<td class="org-right">8</td>
<td class="org-right">1.18356</td>
<td class="org-right">3.26753</td>
</tr>


<tr>
<td class="org-right">10000</td>
<td class="org-right">16</td>
<td class="org-right">1.18217</td>
<td class="org-right">3.02093</td>
</tr>
</tbody>
</table>

