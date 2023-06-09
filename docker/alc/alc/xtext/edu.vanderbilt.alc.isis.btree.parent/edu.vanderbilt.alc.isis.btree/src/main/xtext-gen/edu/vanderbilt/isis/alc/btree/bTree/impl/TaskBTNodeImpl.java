/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree.impl;

import edu.vanderbilt.isis.alc.btree.bTree.BTreePackage;
import edu.vanderbilt.isis.alc.btree.bTree.BehaviorNode;
import edu.vanderbilt.isis.alc.btree.bTree.TaskBTNode;

import java.util.Collection;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Task BT Node</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.impl.TaskBTNodeImpl#getTask <em>Task</em>}</li>
 * </ul>
 *
 * @generated
 */
public class TaskBTNodeImpl extends BTreeNodeImpl implements TaskBTNode
{
  /**
   * The cached value of the '{@link #getTask() <em>Task</em>}' reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getTask()
   * @generated
   * @ordered
   */
  protected EList<BehaviorNode> task;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected TaskBTNodeImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  protected EClass eStaticClass()
  {
    return BTreePackage.Literals.TASK_BT_NODE;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public EList<BehaviorNode> getTask()
  {
    if (task == null)
    {
      task = new EObjectResolvingEList<BehaviorNode>(BehaviorNode.class, this, BTreePackage.TASK_BT_NODE__TASK);
    }
    return task;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object eGet(int featureID, boolean resolve, boolean coreType)
  {
    switch (featureID)
    {
      case BTreePackage.TASK_BT_NODE__TASK:
        return getTask();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @SuppressWarnings("unchecked")
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case BTreePackage.TASK_BT_NODE__TASK:
        getTask().clear();
        getTask().addAll((Collection<? extends BehaviorNode>)newValue);
        return;
    }
    super.eSet(featureID, newValue);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eUnset(int featureID)
  {
    switch (featureID)
    {
      case BTreePackage.TASK_BT_NODE__TASK:
        getTask().clear();
        return;
    }
    super.eUnset(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public boolean eIsSet(int featureID)
  {
    switch (featureID)
    {
      case BTreePackage.TASK_BT_NODE__TASK:
        return task != null && !task.isEmpty();
    }
    return super.eIsSet(featureID);
  }

} //TaskBTNodeImpl
