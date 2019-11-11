'''
Vineet Kumar, sioom.ai
Modify dataset to create new dataset that works with the fairseq code base
From directory examples/dialog/, issue following command:
python3 create-fairseq-dialog-dataset.py data-bin/dialog
'''
import sys
import pathlib
import re
import shutil
import logging
import pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)      # DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)     # DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
formatter = logging.Formatter(
        '%(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# In general, try-except is not used because cannot recover from those failures
tbotDirP = pathlib.Path(sys.argv[0]).parents[0].resolve()   # examples/dialog
baseDirP = tbotDirP.parents[1]  # fairseq base directory
dialogIndexDirP = baseDirP.joinpath(sys.argv[1]).resolve()  # data-bin/dialog
oldDatasetDirP = tbotDirP.joinpath('dialog-bAbI-tasks')
if not oldDatasetDirP.exists():
    strng = (
        f'**Error** Program ended prematurely.\nRun this program after '
        f'downloading the dataset at {oldDatasetDirP}'
    )
    sys.exit(strng)

logger.debug(f'create directories for new dataset')
newDatasetDirP = tbotDirP.joinpath('fairseq-dialog-dataset')
shutil.rmtree(newDatasetDirP, ignore_errors=True)
shutil.rmtree(dialogIndexDirP, ignore_errors=True)
fileNameCmpnts0 = {'task6'}
# fileNameCmpnts0 = {'task1', 'task1-OOV', 'task2', 'task2-OOV', 'task3',
#                   'task3-OOV', 'task4', 'task4-OOV', 'task5',
#                   'task5-OOV', 'task6-dstc2'}
fileNameCmpnts1 = {'dev', 'trn', 'tst'}
fileNameCmpnts1Dict = {'dev': 'valid', 'trn': 'train', 'tst': 'test'}
fileNameCmpnts2 = {'OOV'}
for fileNameCmpnt0 in fileNameCmpnts0:
    newDatasetDirP.joinpath(fileNameCmpnt0).mkdir(parents=True)
    dialogIndexDirP.joinpath(fileNameCmpnt0).mkdir(parents=True)

logger.debug(f'create files for new dataset')
for oldFileP in oldDatasetDirP.glob('dialog-babi-task*.txt'):
    # create bot and hmn files in new dataset from a valid file in old dataset
    oldFileNameCmpnts = re.findall(r'task[1-6]|dev|trn|tst|OOV', oldFileP.stem)
    if len(oldFileNameCmpnts) == 2 and oldFileNameCmpnts[0] in \
            fileNameCmpnts0 and oldFileNameCmpnts[1] in fileNameCmpnts1:
        newBotFile = newDatasetDirP.joinpath(oldFileNameCmpnts[0]).joinpath(
                f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}.bot')
        newHmnFile = newDatasetDirP.joinpath(oldFileNameCmpnts[0]).joinpath(
                f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}.hmn')
        dialogIndexFile = dialogIndexDirP.joinpath(
                oldFileNameCmpnts[0]).joinpath(
                f'dialog-index-{fileNameCmpnts1Dict[oldFileNameCmpnts[1]]}')
    elif len(oldFileNameCmpnts) == 3 and oldFileNameCmpnts[0] in \
            fileNameCmpnts0 and oldFileNameCmpnts[1] in fileNameCmpnts1 \
            and oldFileNameCmpnts[2] in fileNameCmpnts2:
        newBotFile = newDatasetDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(f'\
{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}-{oldFileNameCmpnts[2]}.bot')
        newHmnFile = newDatasetDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(f'\
{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}-{oldFileNameCmpnts[2]}.hmn')
        dialogIndexFile = dialogIndexDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(
                f'dialog-index-{fileNameCmpnts1Dict[oldFileNameCmpnts[1]]}')
    else:
        continue
    newBotFile.touch(exist_ok=False)
    newHmnFile.touch(exist_ok=False)
    dialogIndexFile.touch(exist_ok=False)
    print()
    logger.debug(f'old dataset file: {oldFileP}')
    strng = (
       f'new dataset files: {newBotFile}, {newHmnFile}, {dialogIndexFile}'
    )
    logger.debug(strng)

    logger.debug(f'write data from old dataset file to new dataset files')
    with oldFileP.open('r') as oldFile, newBotFile.open('w') as botFile, \
            newHmnFile.open('w') as hmnFile, \
            dialogIndexFile.open('wb') as dialogFile:
        hmnBotFileLineCount = 0
        dialogStartIndex = []   # record line numbers where dialogs start
        prev_line_apiCall = False
        for lineno, line in enumerate(oldFile):
            if line == '\n':
                continue
            try:
                hmnLine, botLine = line.split('\t')
                prev_line_apiCall = False
            except ValueError as error:
                if prev_line_apiCall:
                    continue
                strng = (
                   f'**Error**: Missing tab separating a human utterance from '
                   f'a bot utterance in the following file, line number, and '
                   f'line:\nFile: {oldFile}\nLine #: {lineno + 1}\nLine: '
                   f'{line}'
                )
                sys.exit(strng)
            if not hmnLine[0].isdecimal():
                strng = (
                   f'**Error**: Missing decimal number at the beginning of '
                   f'the line in the following file, line number, and '
                   f'line:\n File: {oldFile}\n Line #: {lineno + 1}\n Line: '
                   f'{line}'
                )
                sys.exit(strng)
            if hmnLine[0] == '1':
                dialogStartIndex.append(hmnBotFileLineCount)
            hmnFile.write(hmnLine.lstrip('0123456789 ')+'\n')
            botFile.write(botLine)
            if 'api_call' in botLine:
                prev_line_apiCall = True
            hmnBotFileLineCount += 1
        pickle.dump(dialogStartIndex, dialogFile)

# Tests to make sure that there is no problem
    with newHmnFile.open('r') as hmnFile:
        # the sum expression calculates the number of lines in the file;
        # the sum expression is a generator so it cannot be used again unless
        # the file is closed and then opened again
        if hmnBotFileLineCount - sum(1 for _ in hmnFile):
            logger.critical(f'line count is wrong')
    with newHmnFile.open('r') as hmnFile:
        strng = (
           f'file={newHmnFile.stem}: expected line count={hmnBotFileLineCount}'
           f', line count in file={sum(1 for _ in hmnFile)}'
        )
        logger.debug(strng)

    with dialogIndexFile.open('rb') as dialogFile:
        dialogIndexList = pickle.load(dialogFile)
        logger.debug(
                f'dialog index {dialogIndexList[0:5], dialogIndexList[-5:]}')

# cannot do statistics on files because text is not tokenized
