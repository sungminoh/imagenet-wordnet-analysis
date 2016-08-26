# import numpy as np
# from matplotlib import pyplot as plt
from lxml import etree
from collections import Counter
import os
import pickle
from pprint import PrettyPrinter


wordnet_filename = './structure_released_pruned.xml'
# wordnet_filename = './structure_released.xml'
imagenet_filename = './22k_label.txt'
imagenet_wnid_depth_filename = 'imagenet_wnid_depth%s.pickle'\
    % ('_pruned' if 'pruned' in wordnet_filename else '')


class Wordnet(object):
    def __init__(self, tree):
        if type(tree) is str:
            tree = etree.parse(tree)
        self.tree = tree
        self.root = tree.getroot()[1]
        self.misc_root = self.root[-1]
        self.struct_wnids, self.misc_wnids, self.wnids = self.__get_wnids_list(self.root)
        self.struct_wnids_set = set(self.struct_wnids)
        self.misc_wnids_set = set(self.misc_wnids)
        self.wnids_set = set(self.wnids)
        self.tree_string = self.__get_strs()

    def __get_wnids_rec(self, node, attr='wnid'):
        ret = [node.get(attr)]
        for child in node:
            ret.extend(self.__get_wnids_rec(child))
        return ret

    def __get_wnids_list(self, root):
        struct_wnids = []
        for child in root[:8]:
            struct_wnids.extend(self.__get_wnids_rec(child))

        misc_wnids = []
        for child in self.misc_root:
            misc_wnids.extend(self.__get_wnids_rec(child))

        wnids = struct_wnids + misc_wnids
        return struct_wnids, misc_wnids, wnids

    def __generate_strs_rec(self, node, parent_str):
        wnid = node.get('wnid')
        val = node.get('words').split(', ')[0]
        node_str = parent_str + wnid
        output = [node_str + ',' + val]
        for child in node:
            output.extend(self.__generate_strs_rec(child, node_str + '.'))
        return output

    def __get_strs(self):
        if os.path.isfile('flare.csv'):
            return open('flare.csv').readlines()[1:]
        else:
            tree_str = self.__generate_strs_rec(self.root, '')
            with open('flare.csv', 'w') as f:
                f.write('id,value\n' + '\n'.join(tree_str))
            return tree_str


class Imagenet(object):
    def __init__(self, f):
        if type(f) is str:
            f = open(f)
        self.wnids = [line.strip().split()[0] for line in f.readlines()]
        self.wnids_set = set(self.wnids)
        f.close()


class Utils(object):
    def __init__(self, imagenet, wordnet):
        self.imagenet = imagenet
        self.wordnet = wordnet

    def get_node_by_wnid(self, wnid):
        return [node for node in self.wordnet.root.xpath('//synset[@wnid="%s"]' % wnid)]

    def get_wnid_branch_leaf_count_dic(self):
        root = self.wordnet.root
        wnid_branch_leaf_count_dic = {}
        self.__build_wind_branch_leaf_count_dic_rec(root, wnid_branch_leaf_count_dic)
        return wnid_branch_leaf_count_dic

    def get_branch_leaf_count(self):
        wnid_branch_leaf_count_dic = self.get_wnid_branch_leaf_count_dic()
        number_of_leaf_wnid_in_imagenet = 0
        number_of_branch_wnid_in_imagenet = 0
        number_of_both_leaf_and_branch_wnid = 0
        for wnid, dic in wnid_branch_leaf_count_dic.iteritems():
            if dic['branch'] > 0:
                number_of_branch_wnid_in_imagenet += 1
            if dic['leaf'] > 0:
                number_of_leaf_wnid_in_imagenet += 1
                if dic['branch'] > 0:
                    number_of_both_leaf_and_branch_wnid += 1
        return (number_of_branch_wnid_in_imagenet,
                number_of_both_leaf_and_branch_wnid,
                number_of_leaf_wnid_in_imagenet)

    def get_ancestors(self, wnid, attr=None):
        nodes = self.get_node_by_wnid(wnid)
        branches = []
        for node in nodes:
            branches.append(self.__path_string_rec(node, [], attr))
        return branches

    def print_ancestors(self, wnid, attr=None):
        nodes = self.get_node_by_wnid(wnid)
        for node in nodes:
            print '%s has %d childs' % (wnid, len(node)) if len(node) > 0 else '%s is leaf node' % (wnid)
            print '\n'.join(map(str, self.__path_string_rec(node, [], attr)))
            print ''

    def __get_depth(self, wnid_or_node):
        if type(wnid_or_node) == str:
            nodes = self.get_node_by_wnid(wnid_or_node)
        else:
            nodes = [wnid_or_node]
        results = []
        for node in nodes:
            steps_to_leaf = Utils.__count_steps_to_leaf_rec(node)
            steps_to_root = Utils.__count_steps_to_root(node)
            results.append((steps_to_root, steps_to_leaf))
        return results

    def get_depth_all(self, imagenet_wnid_depth_filename='imagenet_wnid_depth_pruned.pickle', refresh=False):
        if not refresh and os.path.isfile(imagenet_wnid_depth_filename):
            with open(imagenet_wnid_depth_filename) as f:
                return pickle.load(f)
        wnid_depth_dic = {}
        for wnid in self.imagenet.wnids:
            wnid_depth_dic[wnid] = self.__get_depth(wnid)
        with open(imagenet_wnid_depth_filename, 'w') as f:
            pickle.dump(wnid_depth_dic, f)
        return wnid_depth_dic

    def get_children_count(self, wnid_or_node):
        if type(wnid_or_node) == str:
            nodes = self.get_node_by_wnid(wnid_or_node)
        else:
            nodes = [wnid_or_node]
        results = []
        for node in nodes:
            results.append(self.__get_children_count_rec(node))
        return results

    def __get_children_count_rec(self, node):
        if len(node) == 0:
            return 1 if node.get('wnid') in self.imagenet.wnids_set else 0
        else:
            return sum(map(self.__get_children_count_rec, node))

    def get_occurence_counter(self, query):
        if query == 'counter_imagenet_wnid_in_wordnet':
            return Counter([wnid for wnid in wordnet.wnids if wnid in imagenet.wnids])
        elif query == 'counter_imagenet_occurence_in_wordnet':
            tmp = Counter([wnid for wnid in wordnet.wnids if wnid in imagenet.wnids])
            return Counter(tmp.values())
        elif query == 'counter_imagenet_wnid_in_struct':
            return Counter([wnid for wnid in wordnet.struct_wnids if wnid in imagenet.wnids])
        elif query == 'counter_imagenet_occurence_in_struct':
            tmp = Counter([wnid for wnid in wordnet.struct_wnids if wnid in imagenet.wnids])
            return Counter(tmp.values())
        return

    def __build_wind_branch_leaf_count_dic_rec(self, node, wnid_branch_leaf_count_dic):
        for child in node:
            self.__build_wind_branch_leaf_count_dic_rec(child, wnid_branch_leaf_count_dic)
        wnid = node.get('wnid')
        if wnid not in self.imagenet.wnids_set:
            return
        if wnid not in wnid_branch_leaf_count_dic:
            wnid_branch_leaf_count_dic[wnid] = dict(branch=0, leaf=0)
        if len(node) == 0:
            wnid_branch_leaf_count_dic[wnid]['leaf'] += 1
        else:
            wnid_branch_leaf_count_dic[wnid]['branch'] += 1
        return

    def prune_wordnet(self, tree=None, refresh=False):
        if not refresh and os.path.isfile('structure_released_pruned.xml'):
            return etree.parse('structure_released_pruned.xml')

        def prune_wordnet_rec(node):
            for child in node:
                prune_wordnet_rec(child)
            if len(node) == 0:
                if node.get('wnid') not in self.imagenet.wnids_set:
                    node.getparent().remove(node)

        if tree is None:
            tree = self.wordnet.tree
        elif type(tree) is str:
            tree = etree.parse('./structure_released.xml')
        root = tree.getroot()[1]
        prune_wordnet_rec(root)
        with open('./structure_released_pruned.xml', 'w') as f:
            tree.write(f)
        return tree

    @staticmethod
    def __path_string_rec(node, stack=[], attr=None):
        if node.getparent() is not None:
            if attr is None:
                stack.append(node.attrib)
                return Utils.__path_string_rec(node.getparent(), stack)
            else:
                stack.append(node.get(attr))
                return Utils.__path_string_rec(node.getparent(), stack, attr)
        else:
            if attr is None:
                stack.append(node.attrib)
                return stack
            else:
                stack.append(node.get(attr))
                return stack

    @staticmethod
    def __count_steps_to_leaf_rec(node):
        if len(node) == 0:
            return 0
        depths = []
        for child in node:
            depths.append(Utils.__count_steps_to_leaf_rec(child) + 1)
        return max(depths)

    @staticmethod
    def __count_steps_to_root(node):
        step = 0
        while node.getparent() is not None:
            node = node.getparent()
            step += 1
        return step


class PrintHelper(object):
    @staticmethod
    def linebreak(size=80, char='='):
        print '\n' + char*size + '\n'

    @staticmethod
    def __print_box(lines):
        box_length = max(map(len, lines)) + 5
        print '-' * box_length
        for line in lines:
            print '|', line.ljust(box_length - 4), '|'
        print '-' * box_length

    @staticmethod
    def barchart(factors, title='', size=50, char='#'):
        lines = []
        if title == '':
            lines.append('bar chart over %d factors' % (len(factors)))
        else:
            lines.append(title)
        if type(factors) == dict:
            factors = list(factors.iteritems())
        max_size = max([factor[1] for factor in factors])
        for key, val in factors:
            bar_string = char * (val*size/max_size)
            lines.insert(1, '{:10s}: [{}] {}'.format(str(key), bar_string, str(val)))
        PrintHelper.__print_box(lines)

    @staticmethod
    def stackchart(factors, title='', size=65, chars=['#', '@']):
        lines = []
        if title == '':
            lines.append('bar chart over %d factors' % (len(factors)))
        else:
            lines.append(title)
        if type(factors) == dict:
            factors = list(factors.iteritems())
        max_size = sum([factor[1] for factor in factors])
        bar_string = ''
        legend_string = ''
        val_string = ''
        for i in range(len(factors)):
            key, val = factors[i]
            char = chars[i % len(chars)]
            bar_string += char * (val*size/max_size)
            legend_string += key.ljust(val*size/max_size)
            val_string += str(val).ljust(val*size/max_size)
        lines.append(bar_string + ' ' + str(max_size))
        lines.append(legend_string)
        lines.append(val_string)
        PrintHelper.__print_box(lines)


def inspect_wordnet(wordnet, imagenet):
    wordnet_intersection = wordnet.struct_wnids_set.intersection(wordnet.misc_wnids_set)
    wordnet_only_struct = wordnet.struct_wnids_set.difference(wordnet.misc_wnids_set)
    wordnet_only_misc = wordnet.misc_wnids_set.difference(wordnet.struct_wnids_set)
    print '#### wordnet wnids ####'
    print 'total:                               %s (unique: %s)'\
        % (len(wordnet.wnids), len(wordnet.wnids_set))
    print 'struct:                              %s (unique: %s)'\
        % (len(wordnet.struct_wnids), len(wordnet.struct_wnids_set))
    print 'misc:                                %s (unique: %s)'\
        % (len(wordnet.misc_wnids), len(wordnet.misc_wnids_set))
    print '%s wnids are in intersection of structured branches and misc'\
        % (len(wordnet_intersection))
    PrintHelper.barchart(dict(struct    = len(wordnet.struct_wnids),
                              misc      = len(wordnet.misc_wnids)),
                         'Number of wnids')
    PrintHelper.barchart(dict(struct    = len(wordnet.struct_wnids_set),
                              misc      = len(wordnet.misc_wnids_set)),
                         'Number of unique wnids')
    PrintHelper.stackchart([('struct'   , len(wordnet_only_struct)),
                            ('inter'    , len(wordnet_intersection)),
                            ('misc'     , len(wordnet_only_misc))],
                           'Wordnet wnid distribution')
    wordnet_subtract_imagenet_set = wordnet.wnids_set.difference(imagenet.wnids_set)
    print 'in wordnet, but not in imagenet:     %s (unique: %s)'\
        % (len([wnid for wnid in wordnet.wnids if wnid in wordnet_subtract_imagenet_set]),
           len(wordnet_subtract_imagenet_set))


def inspect_imagenet(wordnet, imagenet):
    pp = PrettyPrinter(indent=4)
    utils = Utils(imagenet, wordnet)

    wordnet_intersection = wordnet.struct_wnids_set.intersection(wordnet.misc_wnids_set)
    imagenet_only_struct = imagenet.wnids_set.difference(wordnet.misc_wnids_set)
    imagenet_only_misc = imagenet.wnids_set.difference(wordnet.struct_wnids_set)
    imagenet_intersection = imagenet.wnids_set.intersection(wordnet_intersection)
    print '#### imagenet wnids ####'
    print 'number of wnids:                     %s'\
        % (len(imagenet.wnids))
    print 'in imagenet but not in wordet:       %s'\
        % (len(imagenet.wnids_set.difference(wordnet.wnids_set)))
    print 'only in structured:                  %s'\
        % (len(imagenet_only_struct))
    print 'only in misc:                        %s'\
        % (len(imagenet_only_misc))
    PrintHelper.stackchart([('struct'  , len(imagenet_only_struct)),
                            ('inter'   , len(imagenet_intersection)),
                            ('misc'    , len(imagenet_only_misc))],
                           'Imagenet wnid distribution in wordnet')

    branch_leaf_count = utils.get_branch_leaf_count()
    print 'Number of branch wnid in imagenet:   %s' % branch_leaf_count[0]
    print 'Number of both leaf and branch wnid: %s' % branch_leaf_count[1]
    print 'Number of leaf wnid in imagenet:     %s' % branch_leaf_count[2]
    PrintHelper.stackchart(zip(['branch', 'both', 'leaf'], branch_leaf_count),
                           'Imagenet wnid distribution in wordnet 2')

    wnid_depth_dic = utils.get_depth_all(imagenet_wnid_depth_filename)
    threshold_step = 8
    filtered_wnid_depth_dic = {key + ' ' + utils.get_node_by_wnid(key)[0].get('words').split(', ')[0]:
                               map(lambda tup: dict(steps_to_root=tup[0][0],
                                                    steps_to_leaf=tup[0][1],
                                                    sub_imagenet_nodes=tup[1][0]),
                                   zip(val, map(utils.get_children_count, utils.get_node_by_wnid(key))))
                               for key, val in wnid_depth_dic.iteritems()
                               if filter(lambda x: x[1] > threshold_step, val)}
    print 'wnids which have deep subtree (steps to leaf > %d)' % (threshold_step)
    pp.pprint(filtered_wnid_depth_dic)

    # filtered_wnid_depth_dic = {key + ' ' + utils.get_node_by_wnid(key)[0].get('words').split(', ')[0]:
                               # map(lambda tup: dict(steps_to_root=tup[0][0],
                                                    # steps_to_leaf=tup[0][1],
                                                    # sub_imagenet_nodes=tup[1][0]),
                                   # zip(val, map(utils.get_children_count, utils.get_node_by_wnid(key))))
                               # for key, val in wnid_depth_dic.iteritems()
                               # if filter(lambda x: x[1] == 1, val)}
    # filtered_wnid_depth_dic = {key + ' ' + utils.get_node_by_wnid(key)[0].get('words').split(', ')[0]:
                               # map(lambda tup: dict(steps_to_root=tup[0], steps_to_leaf=tup[1]), val)
                               # for key, val in wnid_depth_dic.iteritems()
                               # if filter(lambda x: x[1] == 1, val)}
    print 'wnids which have deep subtree (steps to leaf == %d)' % (1)
    pp.pprint(filtered_wnid_depth_dic)

    branch_wnids = [key for key, val in utils.get_wnid_branch_leaf_count_dic().iteritems()
                    if val['branch'] > 0]
    print 'branch wnids sample 10'
    pp.pprint(branch_wnids[0:10])

    # print '\nimagenet occurence in wordnet'
    # print utils.get_occurence_counter('counter_imagenet_wnid_in_wordnet')
    # print utils.get_occurence_counter('counter_imagenet_occurence_in_wordnet')
    # print utils.get_occurence_counter('counter_imagenet_wnid_in_struct')
    # print utils.get_occurence_counter('counter_imagenet_occurence_in_struct')


wordnet = Wordnet(wordnet_filename)
imagenet = Imagenet(imagenet_filename)
utils = Utils(imagenet, wordnet)


def main():
    imagenet = Imagenet(imagenet_filename)

    """ wordnet """
    inspect_wordnet(wordnet, imagenet)
    PrintHelper.linebreak()

    """ imagenet """
    inspect_imagenet(wordnet, imagenet)
    PrintHelper.linebreak()

    """ ancestors """
    # print 'ancestor inspection\n'
    # print utils.print_ancestors('n00015388', 'words')


if __name__ == '__main__':
    main()
