# mpirun -np 4 python x.py
import sys
sys.path.insert(0, './../..') 
import time
from mpi4py import MPI
import pickle
import unittest

from utils.parallel.distributed import MPIDistributed, MPIModelCastWithMasterSlave



MSG = "message"


class TestMPIDistributed(unittest.TestCase):
    """
    1 learner rank; 2 actor rank; 1 buffer rank
    learner_send: is_block/ buffer / use_req
    actor_recv: is_block / buffer / iprobe
    """
    def setUp(self):
        self.data = MSG
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.mpi_distributed = MPIDistributed(comm=self.comm)
        self.mpi_distributed._learner_rank = [0]
        self.mpi_distributed._actor_rank = [1, 2]
        self.mpi_distributed._buffer_rank = [3]
        self.buff = bytearray(1024)

    def test_learner_send_actor_recv_block(self):
        """
        learner_send is_block, actor_recv is_block
        """
        is_block = True
        
        if self.rank == 0:
            # learner
            self.mpi_distributed.learner_send(self.data, is_block=is_block, use_buffer=False, use_req=False)
        elif self.rank == 1 or self.rank == 2:
            # actor
            result = self.mpi_distributed.actor_recv(self.rank, is_block=is_block)
            self.assertEqual(result, MSG)
        
    def test_learner_send_actor_recv_iprobe(self):
        """
        learner send; actor sleep 1s, iprobe recv
        """
        use_iprobe = True

        if self.rank == 0:
            self.mpi_distributed.learner_send(self.data, is_block=True, use_buffer=False, use_req=False)
        elif self.rank == 1 or self.rank == 2:
            # actor
            while 1:
                result = self.mpi_distributed.actor_recv(self.rank, use_iprobe=use_iprobe)
                if result:
                    self.assertEqual(result, MSG)
                    break

    def test_learner_send_actor_recv_buffer(self):
        """
        learner buffer send; actor buffer recv
        """
        use_buffer = True

        if self.rank == 0:
            dats = pickle.dumps(self.data)
            send_req = self.mpi_distributed.learner_send(dats, use_buffer=use_buffer, use_req=False)
            send_req.Start()
            send_req.Wait()
        elif self.rank == 1 or self.rank == 2:
            time.sleep(1)
            reqs = self.mpi_distributed.actor_recv(self.rank, recv_buffer=self.buff)
            reqs.Start()
            reqs.Wait()
            result = pickle.loads(self.buff)
            self.assertEqual(result, MSG)
            
    def test_learner_send_req_actor_recv_block(self):
        """
        learner_send use_req, actor_recv iprobe
        """
        use_req = True
        use_iprobe = True
        old_req_ls = []
        count = 0
    
        if self.rank == 0:
            while (count < 10):
                count += 1
                old_req_ls = self.mpi_distributed.learner_send(self.data, use_req=use_req, use_timeout=False, old_req_list=old_req_ls)
        elif self.rank == 1 or self.rank == 2:
            while 1:
                result = self.mpi_distributed.actor_recv(self.rank, use_iprobe=use_iprobe)
                if result:
                    self.assertEqual(result, MSG)
                    break


if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestMPIDistributed('test_learner_send_actor_recv_block'))
    t.addTest(TestMPIDistributed('test_learner_send_actor_recv_iprobe'))
    t.addTest(TestMPIDistributed('test_learner_send_actor_recv_buffer'))
    t.addTest(TestMPIDistributed('test_learner_send_req_actor_recv_block'))
    r = unittest.TextTestRunner()
    r.run(t)