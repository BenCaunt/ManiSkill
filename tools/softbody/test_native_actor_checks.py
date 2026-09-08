import unittest
import numpy as np
from verification.native_actor_checks import check_arrays


class NativeActorChecks(unittest.TestCase):
    def data(self):
        initial=np.zeros((4,13),np.float32);initial[:,3]=1
        requested=initial[[3,1]].copy();requested[:,0]+=[.0125,-.0175]
        requested[:,7:10]=[[.13,.24,.35],[-.16,-.27,-.38]]
        after=initial.copy();after[[3,1]]=requested
        data={key:initial.copy() for key in ['initial_public','initial_bridge','initial_requested','assigned','selected_before']}
        data.update({key:after.copy() for key in ['selected_after','selected_bridge','invalid_before','invalid_after']})
        data.update(selection=np.array([3,1]),selected_requested=requested,
                    selected_roundtrips=np.repeat(after[None],21,axis=0),full_roundtrips=np.repeat(after[None],21,axis=0))
        return data

    def test_exact_selection(self):
        self.assertEqual(check_arrays(self.data(),2e-6)['failures'],[])

    def test_unselected_sub_ulp_scale_change_is_not_waived(self):
        data=self.data();data['selected_after'][0,0]=1e-9
        self.assertIn('Selected apply changed an unselected native actor',check_arrays(data,2e-6)['failures'])

    def test_source_index_mixup_is_detected(self):
        data=self.data();data['selected_after'][[3,1]]=data['selected_after'][[1,3]]
        failures=check_arrays(data,2e-6)['failures']
        self.assertIn('Selected velocities differ from request',failures)
        self.assertIn('Selected pose exceeds declared native assignment tolerance',failures)

    def test_pose_roundoff_cannot_hide_velocity_mutation(self):
        data=self.data();data['selected_after'][1,7]+=1e-7
        self.assertIn('Selected velocities differ from request',check_arrays(data,2e-6)['failures'])

    def test_repeated_selected_update_preserves_other_rows_exactly(self):
        data=self.data();data['selected_roundtrips'][8,0,0]=1e-10
        self.assertIn('Repeated selected apply changed unselected actor 0',check_arrays(data,2e-6)['failures'])
