
import math
import itertools


def log2(value):
    return int(math.log2(value))


def sort(inputs):
    length = len(inputs)
    unsorted = list(inputs)

    for major_step in range(0, log2(length)):
        major(unsorted, length, 2**(major_step + 1))

    return unsorted


def major(unsorted, length, block_size):
    for minor_step in range(log2(block_size), 0, -1):
        minor(unsorted, length, block_size, 2**(minor_step - 1))


def minor(unsorted, length, block_size, step_size):
    for micro_step in range(0, step_size):
        micro(unsorted, length, block_size, step_size, micro_step)


def micro(unsorted, length, block_size, step_size, block_offset):
    # print('micro(block_size = %d, step_size = %d, block_offset = %d)' % (block_size, step_size, block_offset))
    for i in range(0, length - step_size, 2 * step_size):
        ascending = (i // block_size) % 2
        compare(unsorted, i + block_offset, i + block_offset + step_size, ascending)


def compare(unsorted, index_a, index_b, ascending):
    # print('compare(%d, %d, %s)' % (index_a, index_b, ['DESC', 'ASC'][ascending]))
    if (ascending):
        if (unsorted[index_a] > unsorted[index_b]):
            swap(unsorted, index_a, index_b)
    else:
        if (unsorted[index_a] < unsorted[index_b]):
            swap(unsorted, index_a, index_b)


def swap(unsorted, index_a, index_b):
    temp = unsorted[index_b]
    unsorted[index_b] = unsorted[index_a]
    unsorted[index_a] = temp


def is_sorted(maybe_sorted):
    for item_src, item_ref in zip(maybe_sorted, sorted(maybe_sorted)[::-1]):
        if item_src != item_ref:
            return False
    return True

for arr in itertools.permutations(range(0, 8)):
    if not is_sorted(sort(arr)):
        print(sort(arr))
