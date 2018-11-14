#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Eric Moyer github.com/epmoyer eric@lemoncrab.com'

from math import log10
from tabulate import tabulate
from to_precision import to_precision

def main():
    """
    Displays a table containing example conversions for to_precision()
    """

    # Build test values
    seed = [float(int(123456789. / 10**x)) for x in range(7, -1, -1)]
    test_values = ([0.0, 1.0, 10.0, 100.0, -1.0] +
                   [x for x in seed] +
                   [x / 10**int(log10(x)) for x in seed] +
                   [x / 10**9  for x in seed])

    option_cases = (
        ('Default (Auto Notation)',                     dict()),
        ('Standard Notation',                           dict(notation='std')),
        ('Engineering Notation',                        dict(notation='eng')),
        ('Scientific Notation',                         dict(notation='sci')),
        ('Standard Notation with zero stripping',       dict(notation='std', strip_zeros=True)),
        ('Scientific Notation with zero stripping',     dict(notation='sci', strip_zeros=True)),
        ('Standard Notation with integer preservation', dict(notation='std', preserve_integer=True)),
        ('Auto Notation with exponent limit of 5',      dict(auto_limit=5)),
    )

    precisions = tuple(range(1, 6))

    # prints out the label, function call, and precision table
    for options_description, options_dict in option_cases:

        '''
        Prints label for table.
        Ex:
        Default (Auto Notation):
            to_precision(value, precision)
        '''
        print(options_description + ':')
        options_string = ', '.join(
            ['value', 'precision'] +
            [note + '=' + repr(inputs) for note, inputs in options_dict.items()])
        print('to_precision({inputs})'.format(inputs=options_string), end='\n' * 2)

        table = []
        for val in test_values:
            table_row = ['{:0.10f}'.format(val).rstrip('0').rstrip('.')]
            for precision in precisions:
                result_string = to_precision(val, precision, **options_dict)
                table_row.append(result_string)
            table.append(table_row)

        headers = ['value'] + ['precision={}'.format(x) for x in precisions]

        print(tabulate(table, headers, disable_numparse=True), end='\n' * 3)

if __name__ == '__main__':
    main()
