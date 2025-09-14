#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a shim to run the data fetching script from the ai_trading_bot package.
"""
import sys
from ai_trading_bot.core.data import main

if __name__ == '__main__':
    sys.exit(main())
