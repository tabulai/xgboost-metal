// Adapted from libc++ 21.1.6 sorting algorithms, LLVM Project contributors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See third_party/llvm/LICENSE.TXT. Changed for Metal device float2 storage,
// explicit bounded stack, and the value-only comparator used by XGBoost.
//
// Equal feature values can carry unequal weights or different zero signs.
// Reproduce the pinned Apple CPU sort permutation only for those batches;
// the usual path uses the parallel Metal merge sort.
inline void sketch_sort_swap(device float2* values, int a, int b) {
  float2 temporary = values[a]; values[a] = values[b]; values[b] = temporary;
}
inline void sketch_sort_three(device float2* values, int a, int b, int c) {
  if (fp32_less(values[b].x, values[a].x)) sketch_sort_swap(values, a, b);
  if (fp32_less(values[c].x, values[b].x)) {
    sketch_sort_swap(values, b, c);
    if (fp32_less(values[b].x, values[a].x)) sketch_sort_swap(values, a, b);
  }
}
inline void sketch_sort_insert(device float2* values, int begin, int end) {
  for (int i = begin + 1; i < end; ++i) {
    float2 value = values[i];
    int j = i;
    while (j > begin && fp32_less(value.x, values[j - 1].x)) {
      values[j] = values[j - 1]; --j;
    }
    values[j] = value;
  }
}
inline bool sketch_sort_incomplete(device float2* values, int begin, int end) {
  if (end - begin <= 5) { sketch_sort_insert(values, begin, end); return true; }
  sketch_sort_three(values, begin, begin + 1, begin + 2);
  uint changes = 0;
  for (int i = begin + 3; i < end; ++i) {
    if (fp32_less(values[i].x, values[i - 1].x)) {
      float2 value = values[i];
      int j = i;
      do { values[j] = values[j - 1]; --j; }
      while (j > begin && fp32_less(value.x, values[j - 1].x));
      values[j] = value;
      if (++changes == 8) return i + 1 == end;
    }
  }
  return true;
}
inline void sketch_heap_sift(device float2* values, int size, int start) {
  if (size < 2 || start > (size - 2) / 2) return;
  int child = 2 * start + 1;
  if (child + 1 < size && fp32_less(values[child].x, values[child + 1].x)) ++child;
  if (fp32_less(values[child].x, values[start].x)) return;
  float2 top = values[start];
  do {
    values[start] = values[child]; start = child;
    if (child > (size - 2) / 2) break;
    child = 2 * child + 1;
    if (child + 1 < size && fp32_less(values[child].x, values[child + 1].x)) ++child;
  } while (!(fp32_less(values[child].x, top.x)));
  values[start] = top;
}
inline void sketch_heap_sort(device float2* values, int size) {
  if (size < 2) return;
  for (int start = (size - 2) / 2; start >= 0; --start) sketch_heap_sift(values, size, start);
  for (int n = size; n > 1; --n) {
    float2 top = values[0];
    int hole = 0;
    do {
      int child = 2 * hole + 1;
      if (child + 1 < n && fp32_less(values[child].x, values[child + 1].x)) ++child;
      values[hole] = values[child]; hole = child;
    } while (hole <= (n - 2) / 2);
    if (hole == n - 1) values[hole] = top;
    else {
      float2 value = values[n - 1];
      values[hole] = value;
      values[n - 1] = top;
      if (hole > 0) {
        int parent = (hole - 1) / 2;
        if (fp32_less(values[parent].x, value.x)) {
          do {
            values[hole] = values[parent]; hole = parent;
            if (!parent) break;
            parent = (parent - 1) / 2;
          } while (fp32_less(values[parent].x, value.x));
          values[hole] = value;
        }
      }
    }
  }
}
inline int sketch_partition_left(device float2* values, int begin, int end) {
  float2 pivot = values[begin];
  int first = begin, last = end;
  if (fp32_less(pivot.x, values[last - 1].x)) {
    do { ++first; } while (!(fp32_less(pivot.x, values[first].x)));
  } else {
    do { ++first; } while (first < last && !(fp32_less(pivot.x, values[first].x)));
  }
  if (first < last) { do { --last; } while (fp32_less(pivot.x, values[last].x)); }
  while (first < last) {
    sketch_sort_swap(values, first, last);
    do { ++first; } while (!(fp32_less(pivot.x, values[first].x)));
    do { --last; } while (fp32_less(pivot.x, values[last].x));
  }
  if (begin != first - 1) values[begin] = values[first - 1];
  values[first - 1] = pivot;
  return first;
}
inline int2 sketch_partition_right(device float2* values, int begin, int end) {
  float2 pivot = values[begin];
  int first = begin, last = end;
  do { ++first; } while (fp32_less(values[first].x, pivot.x));
  if (first == begin + 1) {
    while (first < last) { --last; if (fp32_less(values[last].x, pivot.x)) break; }
  } else { do { --last; } while (!(fp32_less(values[last].x, pivot.x))); }
  bool partitioned = first >= last;
  while (first < last) {
    sketch_sort_swap(values, first, last);
    do { ++first; } while (fp32_less(values[first].x, pivot.x));
    do { --last; } while (!(fp32_less(values[last].x, pivot.x)));
  }
  if (begin != first - 1) values[begin] = values[first - 1];
  values[first - 1] = pivot;
  return int2(first - 1, partitioned);
}
inline void sketch_tie_sort(device float2* values, uint size) {
  if (size < 2) return;
  int4 stack[64];
  uint pending = 0;
  int first = 0, last = int(size), depth = 2 * (31 - int(clz(size)));
  bool leftmost = true;
  while (true) {
    int length = last - first;
    if (length < 24) sketch_sort_insert(values, first, last);
    else if (!depth) sketch_heap_sort(values + first, length);
    else {
      --depth;
      int middle = length / 2;
      if (length > 128) {
        sketch_sort_three(values, first, first + middle, last - 1);
        sketch_sort_three(values, first + 1, first + middle - 1, last - 2);
        sketch_sort_three(values, first + 2, first + middle + 1, last - 3);
        sketch_sort_three(values, first + middle - 1, first + middle, first + middle + 1);
        sketch_sort_swap(values, first, first + middle);
      } else sketch_sort_three(values, first + middle, first, last - 1);
      if (!leftmost && !(fp32_less(values[first - 1].x, values[first].x))) {
        first = sketch_partition_left(values, first, last);
        continue;
      }
      int2 partition = sketch_partition_right(values, first, last);
      int pivot = partition.x;
      if (partition.y) {
        bool left_done = sketch_sort_incomplete(values, first, pivot);
        if (sketch_sort_incomplete(values, pivot + 1, last)) {
          if (!left_done) { last = pivot; continue; }
        } else if (left_done) { first = pivot + 1; continue; }
        else {
          stack[pending++] = int4(pivot + 1, last, depth, 0);
          last = pivot; continue;
        }
      } else {
        stack[pending++] = int4(pivot + 1, last, depth, 0);
        last = pivot; continue;
      }
    }
    if (!pending) break;
    auto next = stack[--pending];
    first = next.x; last = next.y; depth = next.z; leftmost = bool(next.w);
  }
}
