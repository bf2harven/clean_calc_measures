import nibabel as nib

import numpy as np

from skimage import measure

from scipy import ndimage

import copy

import open3d as o3d



class Register_by_tumors:

    def __init__(self, bl_seg, fu_seg, debug=True):

        """

        bl_seg: Baseline segmentaion

        fu_seg: Follwup segmentation

        debug: debug flag to visual 

        """



        self.debug = debug

        xyz_bl = np.argwhere(bl_seg > 0)

        xyz_fu = np.argwhere(fu_seg > 0)





        self.source = o3d.geometry.PointCloud()

        self.source.points = o3d.utility.Vector3dVector(xyz_bl)



        self.target = o3d.geometry.PointCloud()

        self.target.points = o3d.utility.Vector3dVector(xyz_fu)



        # Voxel_size in mm

        self.voxel_size=1



        # init of the data

        self.prepare_dataset()



        # Run Ransac

        self.result_ransac = self.execute_global_registration()



        # Refine registration

        self.result_icp = self.refine_registration()



        # Show results

        if self.debug:

            self.draw_registration_result(self.source, self.target, self.result_icp.transformation)

        

        self.z_bl, self.z_fu, self.middle_z_bl, self.middle_z_fu = self.calculate_z_diff(bl_seg)



        

        



    def calculate_z_diff(self,bl_seg):

        """

        function that calculate the matching Z between 2 segmentation

        """

        largestCC = self.getLargestCC(bl_seg)

        center_of_mass = ndimage.measurements.center_of_mass(largestCC)

        center_of_mass = np.array(center_of_mass)



        biggest_tumor = o3d.geometry.PointCloud()

        biggest_tumor.points = o3d.utility.Vector3dVector([center_of_mass])



        biggest_tumor.transform(self.result_icp.transformation)



        center_of_liver = (np.array(bl_seg.shape)/2).astype(np.int)



        center_of_liver_to = o3d.geometry.PointCloud()

        center_of_liver_to.points = o3d.utility.Vector3dVector([center_of_liver])



        center_of_liver_to.transform(self.result_icp.transformation)



        return int(center_of_mass[2]), int(np.array(biggest_tumor.points)[0][2]), center_of_liver[2], int(np.array(center_of_liver_to.points)[0][2])





    def preprocess_point_cloud(self, pcd, voxel_size):

        """

        preprocessing function that downsample and give the voxels the correct sizes. 

        """

        pcd_down = pcd.voxel_down_sample(voxel_size)



        radius_normal = voxel_size * 2

        pcd_down.estimate_normals(

            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))



        radius_feature = voxel_size * 5

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(

            pcd_down,

            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh





    def prepare_dataset(self):

        """

        prepare the dataset in case we know that the init transformation matrix is different,

        and also preprocessing the point cloud. 

        """

        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],

                                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])



        self.source.transform(trans_init)

        if self.debug:

            self.draw_registration_result(self.source, self.target, np.identity(4))



        self.source_down, self.source_fpfh = self.preprocess_point_cloud(self.source, self.voxel_size)

        self.target_down, self.target_fpfh = self.preprocess_point_cloud(self.target, self.voxel_size)





    def execute_global_registration(self):

        """

        Ransac registration

        """

        distance_threshold = self.voxel_size * 1.5

        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source_down,
            self.target_down,
            self.source_fpfh,
            self.target_fpfh,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
        )





    def refine_registration(self):

        """

        ICP registration

        """

        distance_threshold = self.voxel_size * 0.4



        self.source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(

            radius=0.1, max_nn=30))

        self.target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(

            radius=0.1, max_nn=30))

        return o3d.pipelines.registration.registration_icp(
            self.source,
            self.target,
            distance_threshold,
            self.result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )



    @staticmethod
    def getLargestCC(segmentation):

        """

        semgnetation: segmentation that we will get the biggest connect commponent from.

        """

        labels = measure.label(segmentation)

        return labels == np.argmax(np.bincount(labels.flat)[1:])+1

    

    @staticmethod

    def draw_registration_result(source, target, transformation):

        """

        function to draw the registration results, basiclly for debugging. 

        """

        source_temp = copy.deepcopy(source)

        target_temp = copy.deepcopy(target)

        source_temp.paint_uniform_color([1, 0.706, 0])

        target_temp.paint_uniform_color([0, 0.651, 0.929])

        source_temp.transform(transformation)

        o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    bl_labeled_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Y_22_01_2020_FU_A_Y_04_05_2020/BL_Scan_Tumors_unique_13_CC.nii.gz'
    fu_labeled_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Y_22_01_2020_FU_A_Y_04_05_2020/FU_Scan_Tumors_unique_12_CC.nii.gz'

    # Baseline load

    # bl_path = 'C:\\Users\\Adi\\PycharmProjects\\Liver_segmentation\\temp\\tumors\\case_19_BL__GT.nii.gz'
    bl_path = bl_labeled_tumors_file

    bl = nib.load(bl_path)

    bl_data = bl.get_fdata()



    # Followup load

    # fu_path = 'C:\\Users\\Adi\\PycharmProjects\\Liver_segmentation\\temp\\tumors\\case_19_FU__GT.nii.gz'
    fu_path = fu_labeled_tumors_file

    fu = nib.load(fu_path)

    fu_data = fu.get_fdata()





    # calling and init the class. 



    registered_class = Register_by_tumors(bl_data,fu_data,debug=True)



    # printing the matching slices

    print(
        f"Baseline slice {registered_class.z_bl} match to Followup slice {registered_class.z_fu}"
    )

    print(
        f"Baseline middle_slice {registered_class.middle_z_bl} match to Followup middle_slice {registered_class.middle_z_fu}"
    )



    # Transformation matrix. 

    print("Transformation Matrix is: ")

    print(registered_class.result_icp.transformation)





