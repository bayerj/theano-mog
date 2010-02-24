"""Module that contains Theano extensions which are not in theano itself but 
needed here."""


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


import numpy, scipy.linalg
from theano import gof, tensor, scalar


class Inv(gof.Op):
    """
    Find the inverse of a Matrix. 
    
    Uses scipy.linalg.inv to perform the operation.
    """

    #TODO: Add class options to use the performance-enhancing flags
    #     sym_pos, lower, overwrite_a, overwrite_b

    #TODO: Add C code that calls the underlying LAPACK routines
    #      and keeps a memory workspace from call to call as a non-default Op output

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, A):
        A_ = tensor.as_tensor_variable(A)
        if A_.broadcastable != (False, False):
            raise TypeError("A must be a matrix", A_.type)
        otype = tensor.TensorType(broadcastable=A_.broadcastable, dtype=A_.type.dtype)
        return gof.Apply(op=self, inputs=[A], outputs=[otype()])

    def perform(self, node, (A,), (output, )):
        ret=scipy.linalg.inv(A)
        if ret.dtype != node.outputs[0].dtype:
            print >> sys.stderr, "WARNING: Solve.perform() required cast."
            ret = theano._asarray(ret, dtype=node.outputs[0].dtype)
        output[0]=ret

inv = Inv()


class Det(gof.Op):
    """
    Find the determinant of a Matrix. 
    
    Uses scipy.linalg.det to perform the operation.
    """

    #TODO: Add class options to use the performance-enhancing flags
    #     sym_pos, lower, overwrite_a, overwrite_b

    #TODO: Add C code that calls the underlying LAPACK routines
    #      and keeps a memory workspace from call to call as a non-default Op output

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, A):
        A_ = tensor.as_tensor_variable(A)
        if A_.broadcastable != (False, False):
            raise TypeError("A must be a matrix", A_.type)
        otype = tensor.scalar
        return gof.Apply(op=self, inputs=[A], outputs=[otype()])

    def perform(self, node, (A,), (output, )):
        ret = scipy.asarray(scipy.linalg.det(A))
        if ret.dtype != node.outputs[0].dtype:
            print >> sys.stderr, "WARNING: Det.perform() returned invalid type."
        output[0]=ret

det = Det()
